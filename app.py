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
    , "southpark",
    "centrist"
]

# ✅ Load Pre-trained Models
try:
    sentiment_model = joblib.load('sentiment_forecast_model.pkl')
    logging.info("✅ Sentiment forecast model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading sentiment model: {e}")
    raise Exception("Failed to load sentiment classifier.")

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


cache_time=1
# ✅ Implement Caching for 24 Hours (1 Day)
cache = TTLCache(maxsize=10, ttl=cache_time)  # Cache lasts for 24 hours
last_update_time = None  # Track last update timestamp

# ✅ Fetch Posts Asynchronously
async def fetch_recent_posts(subreddit_name):
    """Fetches posts asynchronously from a subreddit."""
    subreddit = await async_reddit.subreddit(subreddit_name)
    posts = []
    try:
        async for post in subreddit.top(time_filter="month", limit=25):
            posts.append({
                "subreddit": subreddit_name,
                "date": datetime.datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                "post_text": post.title
            })
    except Exception as e:
        logging.error(f"❌ Error fetching posts from r/{subreddit_name}: {e}")
    return posts

# ✅ Fetch All Posts in Parallel
async def fetch_all_posts():
    """Fetch posts from all subreddits concurrently."""
    tasks = [fetch_recent_posts(sub) for sub in SUBREDDITS]
    results = await asyncio.gather(*tasks)
    all_posts = [post for result in results for post in result]  # Flatten list
    return all_posts

# ✅ Predict Sentiment Score
def predict_score(text):
    """Predicts sentiment score using the PyTorch model."""
    if not text:
        return 0.0
    encoded_input = tokenizer(text.split(), return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        score = score_model(encoded_input["input_ids"], encoded_input["attention_mask"])[0].item()
    return score

# ✅ Generate Sentiment Forecast (Updated to Run Once Per Day)
async def generate_forecast():
    """Generates forecast by analyzing subreddit posts once per day."""
    global last_update_time

    # ✅ Check if it's been 24 hours since the last update
    if "forecast" in cache and last_update_time and (time.time() - last_update_time < cache_time):
        logging.info("✅ Using cached forecast (less than 24 hours old).")
        return cache["forecast"]

    logging.info("🔄 Generating new forecast (more than 24 hours since last update)...")
    
    start_time = datetime.datetime.utcnow() - datetime.timedelta(days=14)
    all_posts = await fetch_all_posts()

    if not all_posts:
        logging.warning("⚠️ No posts fetched! Using fallback random values.")
        # df = pd.DataFrame({
        #     "date_only": [start_time.date() + datetime.timedelta(days=i) for i in range(14)],
        #     "sentiment_score": np.random.uniform(0.5, 0.6, 14)
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

    # ✅ Update cache and timestamp
    cache["forecast"] = pred
    last_update_time = time.time()
    
    return pred

# ✅ Generate Graph with Smooth Curves
async def generate_graph():
    pred = await generate_forecast()

    # ✅ Generate X-axis labels
    today = datetime.date.today()
    days = [today + datetime.timedelta(days=i) for i in range(7)]
    days_str = [day.strftime('%a %m/%d') for day in days]

    # ✅ Generate smooth curve using interpolation
    x = np.arange(7)
    y = np.array(pred)
    
    x_smooth = np.linspace(x.min(), x.max(), 300)  # Smooth X-axis
    spline = make_interp_spline(x, y, k=3)  # Cubic smoothing
    y_smooth = spline(x_smooth)

    # ✅ Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x_smooth, y_smooth, color='#244B48', alpha=0.4)
    ax.plot(x_smooth, y_smooth, color='#244B48', lw=3, label='Forecast')  # Smooth line
    ax.scatter(x, y, color='#244B48', s=100, zorder=5)  # Keep original points

    ax.set_title("7-Day Political Sentiment Forecast", fontsize=16, fontweight='bold')
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Negative Sentiment (0-1)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(days_str, fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return img

# ✅ FastAPI Setup
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Sentiment Forecast API is running!"}

@app.get("/graph.png")
async def get_graph():
    img = await generate_graph()
    return Response(content=img.getvalue(), media_type="image/png")
