import gradio as gr
import time
import datetime
import numpy as np
import pandas as pd
import praw
import joblib
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for Render
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import torch
import scipy.sparse as sp
import torch.nn as nn
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from transformers import AutoTokenizer
import asyncpraw
import asyncio
from fastapi import FastAPI, HTTPException
from starlette.responses import Response
import matplotlib.font_manager as fm
from fastapi import FastAPI, Response
import io
import logging
from cachetools import TTLCache
import time
import base64

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
    'politics','humor']

# ✅ Load Pre-trained Models
sentiment_model = joblib.load("models/sentiment_forecast_model.pkl")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/xlm-twitter-politics-sentiment")
# Load models and data (your existing code)
autovectorizer = joblib.load('AutoVectorizer.pkl')
autoclassifier = joblib.load('AutoClassifier.pkl')
MODEL = "cardiffnlp/xlm-twitter-politics-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Define the ScorePredictor model
class ScorePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1):
        super().__init__()
        super(ScorePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        final_hidden_state = lstm_out[:, -1, :]
        output = self.fc(final_hidden_state)
        return self.sigmoid(output)

# ✅ Load PyTorch Model
score_model = ScorePredictor(tokenizer.vocab_size)
score_model.load_state_dict(torch.load("models/score_predictor.pth", map_location=torch.device("cpu")))
score_model.load_state_dict(torch.load("score_predictor.pth"))
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
sentiment_model = joblib.load('sentiment_forecast_model.pkl')

reddit = praw.Reddit(
    client_id="PH99oWZjM43GimMtYigFvA",
    client_secret="3tJsXQKEtFFYInxzLEDqRZ0s_w5z0g",
    user_agent='MyAPI/0.0.1',
    check_for_async=False
)

subreddits = [
    "centrist",
    "libertarian",
    "southpark",
    "truechristian",
    "conservatives"
]

# Global variable for the generated plot (as a base64-encoded PNG)
prediction_plot_base64 = None

def process_data():
    """Fetches data, performs analysis, and generates the forecast plot."""
    global prediction_plot_base64
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=14)

    def fetch_all_recent_posts(subreddit_name, start_time, limit=500):
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        try:
            for post in subreddit.top(limit=limit):
                post_time = datetime.datetime.utcfromtimestamp(post.created_utc)
                if post_time >= start_time:
                    posts.append({
                        "subreddit": subreddit_name,
                        "timestamp": post.created_utc,
                        "date": post_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "post_text": post.title
                    })
        except Exception as e:
            print(f"Error fetching posts from r/{subreddit_name}: {e}")
        return posts

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict_score(text):
        if not text:
            return 0.0
        max_length = 512
        encoded_input = tokenizer(
            text.split(),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
        input_ids, attention_mask = encoded_input["input_ids"], encoded_input["attention_mask"]
        with torch.no_grad():
            score = score_model(input_ids, attention_mask)[0].item()
        return score

    start_time = datetime.datetime.utcnow() - datetime.timedelta(days=14)
    all_posts = []
    for sub in subreddits:
        print(f"Fetching posts from r/{sub}")
        posts = fetch_all_recent_posts(sub, start_time)
        all_posts.extend(posts)
        print(f"Fetched {len(posts)} posts from r/{sub}")

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

    daily_sentiment = df.groupby('date_only')['sentiment_score'].median()
    daily_sentiment = daily_sentiment.interpolate().fillna(method='bfill').fillna(method='ffill')
    last_14_dates = df['date_only'].unique()
    num_dates = min(len(last_14_dates), 14)
    last_14_dates = sorted(last_14_dates, reverse=True)[:num_dates]

    sentiment_scores_np = daily_sentiment.values.reshape(1, -1)
    pred = sentiment_model.predict(sentiment_scores_np)[0]

    cache["forecast"] = pred
    last_update_time = time.time()
    return pred
    filtered_df = df[df['date_only'].isin(last_14_dates)]
    daily_sentiment = filtered_df.groupby('date_only')['sentiment_score'].median()

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
    if len(daily_sentiment) < 14:
        mean_sentiment = daily_sentiment.mean()
        padding = [mean_sentiment] * (14 - len(daily_sentiment))
        daily_sentiment = np.concatenate([daily_sentiment.values, padding])
        daily_sentiment = pd.Series(daily_sentiment)

    sentiment_scores_np = daily_sentiment.values.reshape(1, -1)
    prediction = sentiment_model.predict(sentiment_scores_np)
    pred = prediction[0]

    # Load a custom font if available
    font_path = "AfacadFlux-VariableFont_slnt,wght[1].ttf"
    custom_font = fm.FontProperties(fname=font_path)

    today = datetime.date.today()
    days = [today + datetime.timedelta(days=i) for i in range(7)]
    days_str = [day.strftime('%a %m/%d') for day in days]

    xnew = np.linspace(0, 6, 300)
    spline = make_interp_spline(np.arange(7), pred, k=3)
    pred_smooth = spline(xnew)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(xnew, pred_smooth, color='#244B48', alpha=0.4)
    ax.plot(xnew, pred_smooth, color='#244B48', lw=3, label='Forecast')
    ax.scatter(np.arange(7), pred, color='#244B48', s=100, zorder=5)

    ax.set_title("7-Day Political Sentiment Forecast", fontsize=22, fontweight='bold', pad=20, fontproperties=custom_font)
    ax.set_xlabel("Day", fontsize=16, fontproperties=custom_font)
    ax.set_ylabel("Negative Sentiment (0-1)", fontsize=16, fontproperties=custom_font)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(days_str, fontsize=14, fontproperties=custom_font)
    ax.set_yticklabels([f"{tick:.2f}" for tick in ax.get_yticks()], fontsize=14, fontproperties=custom_font)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(fontsize=14, loc='upper right', prop=custom_font)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    prediction_plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    print("Data processing complete and plot updated.")

def display_plot():
    global prediction_plot_base64
    if prediction_plot_base64:
        return f'<img src="data:image/png;base64,{prediction_plot_base64}" alt="Prediction Plot">'
    else:
        return "Processing data..."

# Initial data processing (this can be triggered by your cron job)
process_data()

# FastAPI Integration
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Sentiment Forecast API is running!"}

async def generate_graph():
    global prediction_plot_base64
    # Ensure the plot is generated before returning it
    if prediction_plot_base64 is None:
        process_data()
    img_bytes = base64.b64decode(prediction_plot_base64)
    buffer = io.BytesIO(img_bytes)
    return buffer

@app.get("/graph.png")
async def get_graph():
    img = await generate_graph()
    return Response(content=img.getvalue(), media_type="image/png")


async def generate_graph():
    global prediction_plot_base64
    # Ensure the plot is generated before returning it
    if prediction_plot_base64 is None:
        process_data()
    img_bytes = base64.b64decode(prediction_plot_base64)
    buffer = io.BytesIO(img_bytes)
    return buffer

@app.get("/graph.png")
async def get_graph():
    img = await generate_graph()
    return Response(content=img.getvalue(), media_type="image/png")
