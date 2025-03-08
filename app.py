import os
import datetime
import joblib
import numpy as np
import pandas as pd
import torch
import praw
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from torch import nn
from fastapi import FastAPI, HTTPException
from starlette.responses import Response
import io
import requests
import logging
from transformers import AutoTokenizer

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Load Reddit API Credentials (Use environment variables)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "MyAPI/0.0.1")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")

if not all([REDDIT_CLIENT_ID, REDDIT_SECRET, REDDIT_USER_AGENT]):
    logging.error("❌ Reddit API credentials are missing! Check environment variables.")
    raise Exception("Reddit API credentials not set.")

# ✅ Initialize Reddit API
try:
    if REDDIT_USERNAME and REDDIT_PASSWORD:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            username=REDDIT_USERNAME,
            password=REDDIT_PASSWORD,
            check_for_async=False
        )
    else:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            check_for_async=False
        )
    logging.info("✅ Reddit API initialized!")
except Exception as e:
    logging.error(f"❌ Error initializing Reddit API: {e}")
    raise Exception("Reddit API failed to initialize.")

# ✅ Subreddits to monitor
SUBREDDITS = [
    "florida", "ohio", "libertarian", "southpark",
    "walkaway", "truechristian", "conservatives"
]

# ✅ Load Pre-trained Models
try:
    autovectorizer = joblib.load('AutoVectorizer.pkl')
    autoclassifier = joblib.load('AutoClassifier.pkl')
    sentiment_model = joblib.load('sentiment_forecast_model.pkl')
    logging.info("✅ Pre-trained vectorizer & classifier loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading vectorizer/classifier: {e}")
    raise Exception("Failed to load sentiment classifier.")

# ✅ Load Sentiment Model
MODEL = "cardiffnlp/xlm-twitter-politics-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

class ScorePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1):
        super(ScorePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        final_hidden_state = lstm_out[:, -1, :]
        output = self.fc(final_hidden_state)
        return self.sigmoid(output)

score_model = ScorePredictor(tokenizer.vocab_size)
try:
    score_model.load_state_dict(torch.load("score_predictor.pth", map_location=torch.device('cpu')))
    score_model.eval()
    logging.info("✅ Sentiment model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading sentiment model: {e}")
    raise Exception("Sentiment model failed to load.")

# ✅ Fetch Posts from Reddit
def fetch_all_recent_posts(subreddit_name, start_time, limit=500):
    """Fetches posts from Reddit and filters them by time."""
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    try:
        for post in subreddit.new(limit=limit):
            post_time = datetime.datetime.utcfromtimestamp(post.created_utc)
            if post_time >= start_time:
                posts.append({
                    "subreddit": subreddit_name,
                    "timestamp": post.created_utc,
                    "date": post_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "post_text": post.title
                })
    except Exception as e:
        logging.error(f"❌ Error fetching posts from r/{subreddit_name}: {e}")
    return posts

# ✅ Predict Sentiment
def predict_score(text):
    """Predicts sentiment score using the PyTorch model."""
    if not text:
        return None
    encoded_input = tokenizer(
        text.split(),
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids, attention_mask = encoded_input["input_ids"], encoded_input["attention_mask"]
    with torch.no_grad():
        score = score_model(input_ids, attention_mask)[0].item()
    return score

# ✅ Start Data Collection
start_time = datetime.datetime.utcnow() - datetime.timedelta(days=14)
all_posts = []
for sub in SUBREDDITS:
    logging.info(f"Fetching posts from r/{sub}...")
    posts = fetch_all_recent_posts(sub, start_time)
    all_posts.extend(posts)
    logging.info(f"✅ Fetched {len(posts)} posts from r/{sub}")

if not all_posts:
    logging.warning("⚠️ No posts fetched! Check Reddit API credentials.")

# ✅ Handle Empty DataFrame
if not all_posts:
    logging.error("❌ No data fetched. Using random fallback values.")
    df = pd.DataFrame({
        "date_only": [start_time.date() + datetime.timedelta(days=i) for i in range(14)],
        "sentiment_score": np.random.uniform(0.3, 0.7, 14)
    })
else:
    df = pd.DataFrame(all_posts)
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    df = df.sort_values(by=['date_only'])
    df['sentiment_score'] = df['post_text'].apply(predict_score)
    df = df.dropna()

# ✅ Aggregate Sentiment by Day
daily_sentiment = df.groupby('date_only')['sentiment_score'].median()

# ✅ Fill Missing Dates
expected_dates = pd.date_range(start=start_time.date(), periods=14)
daily_sentiment = daily_sentiment.reindex(expected_dates, fill_value=np.nan)
daily_sentiment = daily_sentiment.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# ✅ Forecast Future Sentiment
sentiment_scores_np = daily_sentiment.values.reshape(1, -1)
pred = sentiment_model.predict(sentiment_scores_np)[0]

# ✅ Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "Sentiment Forecast API is running!"}

@app.get("/graph.png")
def generate_graph():
    """Generate and return a sentiment forecast graph."""
    try:
        img = io.BytesIO()  # ✅ FIXED: Ensure img is initialized
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(range(7), pred, color='#244B48', alpha=0.4)
        ax.plot(range(7), pred, color='#244B48', lw=3, label='Forecast')
        ax.scatter(range(7), pred, color='#244B48', s=100, zorder=5)
        plt.savefig(img, format='png')
        img.seek(0)
        return Response(content=img.getvalue(), media_type="image/png")
    except Exception as e:
        logging.error(f"❌ Failed to generate graph: {e}")
        raise HTTPException(status_code=500, detail="Error generating graph")
