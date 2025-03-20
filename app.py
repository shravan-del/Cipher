import os
import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for Render
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import asyncpraw
import asyncio
from fastapi import FastAPI, HTTPException
from starlette.responses import Response
import io
import logging
from cachetools import TTLCache
import time
import re

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Load Reddit API Credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "D9IRrBYtJO37pc7Xgimq6g")
REDDIT_SECRET = os.getenv("REDDIT_SECRET", "iRiiXDqxTfHuMiAOKaxsXEoEPeJfHA")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "MyRedditApp/0.1 by Shravan")

if not all([REDDIT_CLIENT_ID, REDDIT_SECRET, REDDIT_USER_AGENT]):
    logging.error("❌ Reddit API credentials are missing!")
    raise Exception("Reddit API credentials not set.")

# ✅ Initialize Async PRAW (Asynchronous Reddit API)
async_reddit = asyncpraw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# ✅ Subreddits to monitor
SUBREDDITS = [
    "politics"
    # "florida",
    # "ohio",
    # "libertarian",
    # "southpark",
    # "walkaway",
    # "truechristian",
    # "conservatives"
]

# ✅ Load Pre-trained Models
try:
    autovectorizer = joblib.load('AutoVectorizer.pkl')
    autoclassifier = joblib.load('AutoClassifier.pkl')
    sentiment_model = joblib.load('sentiment_forecast_model.pkl')
    logging.info("✅ Models loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading models: {e}")
    raise Exception("Failed to load models.")

# ✅ Load Sentiment Model
MODEL = "cardiffnlp/xlm-twitter-politics-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

class ScorePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

# ✅ Load PyTorch Model
score_model = ScorePredictor(tokenizer.vocab_size)
try:
    score_model.load_state_dict(torch.load("score_predictor.pth", map_location=torch.device('cpu')))
    score_model.eval()
    logging.info("✅ Sentiment model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading sentiment model: {e}")
    raise Exception("Sentiment model failed to load.")

cache_time = 86400  # 24 Hours
# ✅ Implement Caching for 24 Hours (1 Day)
cache = TTLCache(maxsize=10, ttl=cache_time)  # Cache lasts for 24 hours
last_update_time = None  # Track last update timestamp

# ✅ Fetch Posts Asynchronously
async def fetch_recent_posts(subreddit_name, start_time, limit=500):
    subreddit = await async_reddit.subreddit(subreddit_name)
    posts = []
    try:
        async for post in subreddit.new(limit=limit):
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

# ✅ Preprocess Text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ✅ Predict Sentiment Score
def predict_score(text):
    if not text:
        return 0.0
    encoded_input = tokenizer(text.split(), return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        score = score_model(encoded_input["input_ids"], encoded_input["attention_mask"])[0].item()
    return score

# ✅ Generate Sentiment Forecast (Updated to Run Once Per Day)
async def generate_forecast():
    global last_update_time

    if "forecast" in cache and last_update_time and (time.time() - last_update_time < cache_time):
        logging.info("✅ Using cached forecast (less than 24 hours old).")
        return cache["forecast"]

    logging.info("🔄 Generating new forecast (more than 24 hours since last update)...")

    start_time = datetime.datetime.utcnow() - datetime.timedelta(days=14)
    all_posts = []

    tasks = [fetch_recent_posts(sub, start_time) for sub in SUBREDDITS]
    results = await asyncio.gather(*tasks)
    all_posts = [post for result in results for post in result]

    filtered_posts = []
    for post in all_posts:
        vector = autovectorizer.transform([post['post_text']])
        prediction = autoclassifier.predict(vector)
        if prediction[0] == 1:
            filtered_posts.append(post)
    all_posts = filtered_posts

    df = pd.DataFrame(all_posts)
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    df = df.sort_values(by=['date_only'])
    df['sentiment_score'] = df['post_text'].apply(predict_score)

    last_14_dates = df['date_only'].unique()
    num_dates = min(len(last_14_dates), 14)
    last_14_dates = sorted(last_14_dates, reverse=True)[:num_dates]

    filtered_df = df[df['date_only'].isin(last_14_dates)]
    daily_sentiment = filtered_df.groupby('date_only')['sentiment_score'].median()

    if len(daily_sentiment) < 14:
        mean_sentiment = daily_sentiment.mean()
        padding = [mean_sentiment] * (14 - len(daily_sentiment))
        daily_sentiment = np.concatenate([daily_sentiment.values, padding])
        daily_sentiment = pd.Series(daily_sentiment)

    sentiment_scores_np = daily_sentiment.values.reshape(1, -1)
    pred = sentiment_model.predict(sentiment_scores_np)[0]

    cache["forecast"] = pred
    last_update_time = time.time()

    return pred

# ✅ Generate Graph with Smooth Curves
async def generate_graph():
    pred = await generate_forecast()

    today = datetime.date.today()
    days = [today + datetime.timedelta(days=i) for i in

# ✅ FastAPI Setup
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Sentiment Forecast API is running!"}

@app.get("/graph.png")
async def get_graph():
    img = await generate_graph()
    return Response(content=img.getvalue(), media_type="image/png")
