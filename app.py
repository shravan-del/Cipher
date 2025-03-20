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
    logging.error("❌ Missing Reddit API credentials!")
    raise Exception("Reddit API credentials not set.")

# ✅ Initialize Async PRAW
async_reddit = asyncpraw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# ✅ Subreddits to monitor
SUBREDDITS = ["centrist",
    "southpark",
    "truechristian",
    'politics']

# ✅ Load Pre-trained Models
sentiment_model = joblib.load("models/sentiment_forecast_model.pkl")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/xlm-twitter-politics-sentiment")

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
score_model.load_state_dict(torch.load("models/score_predictor.pth", map_location=torch.device("cpu")))
score_model.eval()

#cache = TTLCache(maxsize=10, ttl=86400)  # Cache lasts for 24 hours
last_update_time = None  # Track last update timestamp

async def fetch_recent_posts(subreddit_name):
    """Fetches recent posts asynchronously."""
    subreddit = await async_reddit.subreddit(subreddit_name)
    posts = []
    async for post in subreddit.top(time_filter="month", limit=25):
        posts.append({
            "subreddit": subreddit_name,
            "date": datetime.datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            "post_text": post.title
        })
    return posts

async def fetch_all_posts():
    """Fetches posts from all subreddits concurrently."""
    tasks = [fetch_recent_posts(sub) for sub in SUBREDDITS]
    results = await asyncio.gather(*tasks)
    return [post for result in results for post in result]

def predict_score(text):
    """Predicts sentiment score using PyTorch model."""
    if not text:
        return 0.0
    encoded_input = tokenizer(text.split(), return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        score = score_model(encoded_input["input_ids"], encoded_input["attention_mask"])[0].item()
    return score

async def generate_forecast():
    """Generates sentiment forecast."""
    global last_update_time

    # if "forecast" in cache and last_update_time and (time.time() - last_update_time < 86400):
    #     return cache["forecast"]

    all_posts = await fetch_all_posts()
    df = pd.DataFrame(all_posts)
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    df['sentiment_score'] = df['post_text'].apply(predict_score)

    daily_sentiment = df.groupby('date_only')['sentiment_score'].median()
    daily_sentiment = daily_sentiment.interpolate().fillna(method='bfill').fillna(method='ffill')

    sentiment_scores_np = daily_sentiment.values.reshape(1, -1)
    pred = sentiment_model.predict(sentiment_scores_np)[0]

    cache["forecast"] = pred
    last_update_time = time.time()
    return pred

async def generate_graph():
    """Generates and returns the forecast graph."""
    pred = await generate_forecast()
    x = np.arange(7)
    y = np.array(pred)

    x_smooth = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x_smooth, y_smooth, color='#244B48', alpha=0.4)
    ax.plot(x_smooth, y_smooth, color='#244B48', lw=3, label='Forecast')
    ax.scatter(x, y, color='#244B48', s=100, zorder=5)

    ax.set_title("7-Day Political Sentiment Forecast")
    ax.set_xlabel("Day")
    ax.set_ylabel("Negative Sentiment (0-1)")
    ax.legend()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return img

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Sentiment Forecast API is running!"}

@app.get("/graph.png")
async def get_graph():
    img = await generate_graph()
    return Response(content=img.getvalue(), media_type="image/png")
