import os
import datetime
import praw
import joblib
import torch
import torch.nn as nn
import pandas as pd
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.interpolate import make_interp_spline
from transformers import AutoTokenizer
import pytz
import io
import base64
from fastapi import FastAPI
from starlette.responses import Response
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load models and data
logging.info("Loading models...")
autovectorizer = joblib.load('AutoVectorizer.pkl')
autoclassifier = joblib.load('AutoClassifier.pkl')
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
score_model.load_state_dict(torch.load("score_predictor.pth"))
score_model.eval()

sentiment_model = joblib.load('sentiment_forecast_model.pkl')

# Load Reddit API Credentials from environment variables
# REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
# REDDIT_SECRET = os.getenv("REDDIT_SECRET", "")
# REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "MyAPI/0.0.1")


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

# Cache for the generated graph
cache = {
    "image_data": None,
    "last_updated": None
}

def fetch_all_recent_posts(subreddit_name, start_time, limit=500):
    """Fetch recent posts from a subreddit."""
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
        logging.error(f"Error fetching posts from r/{subreddit_name}: {e}")

    return posts

def preprocess_text(text):
    """Clean and preprocess text data."""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_score(text):
    """Predict sentiment score for a given text."""
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

# some key logic: if today's date is after that last update and the time is after 1:00 am then allow update if not return false
def should_update():
    # sets that timezone
    est = pytz.timezone('America/New_York')
    # fetches that data and keeps it in that eastern time
    now = datetime.datetime.now(est)
    # now with this last_updated cache I am returning true to trigger that update
    if not cache["last_updated"]:
        return True
    # converts the previous one to est as well
    last = cache["last_updated"].astimezone(est)
    return now.date() > last.date() and now.hour >= 1


    
def process_data():
    """Fetches data, performs analysis, and generates the plot."""
    logging.info("Starting data processing...")
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=14)
    start_time = datetime.datetime.utcnow() - datetime.timedelta(days=14)
    
    all_posts = []
    for sub in subreddits:
        logging.info(f"Fetching posts from r/{sub}")
        posts = fetch_all_recent_posts(sub, start_time)
        all_posts.extend(posts)
        logging.info(f"Fetched {len(posts)} posts from r/{sub}")

    logging.info("Filtering political posts...")
    filtered_posts = []
    for post in all_posts:
        vector = autovectorizer.transform([post['post_text']])
        prediction = autoclassifier.predict(vector)
        if prediction[0] == 1:
            filtered_posts.append(post)
    all_posts = filtered_posts
    logging.info(f"Filtered to {len(filtered_posts)} political posts")

    df = pd.DataFrame(all_posts)
    if df.empty:
        logging.error("No posts found for analysis")
        return None
        
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    df = df.sort_values(by=['date_only'])
    
    logging.info("Calculating sentiment scores...")
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

    logging.info("Generating forecast...")
    sentiment_scores_np = daily_sentiment.values.reshape(1, -1)
    prediction = sentiment_model.predict(sentiment_scores_np)
    pred = prediction[0]

    font_path = "AfacadFlux-VariableFont_slnt,wght[1].ttf"
    try:
        custom_font = fm.FontProperties(fname=font_path)
    except Exception as e:
        logging.warning(f"Custom font not found: {e}. Using default font.")
        custom_font = None

    today = datetime.date.today()
    days = [today + datetime.timedelta(days=i) for i in range(7)]
    days_str = [day.strftime('%a %m/%d') for day in days]

    xnew = np.linspace(0, 6, 300)
    spline = make_interp_spline(np.arange(7), pred, k=3)
    pred_smooth = spline(xnew)

    logging.info("Creating visualization...")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(xnew, pred_smooth, color='#244B48', alpha=0.4)
    ax.plot(xnew, pred_smooth, color='#244B48', lw=3, label='Forecast')
    ax.scatter(np.arange(7), pred, color='#244B48', s=100, zorder=5)
    
    est_timezone = pytz.timezone('America/New_York')
    est_time = datetime.datetime.now(est_timezone)
    ax.set_title(f"7-Day Political Sentiment Forecast - {est_time.strftime('%Y-%m-%d %H:%M:%S EST')}", 
             fontsize=22, fontweight='bold', pad=20, fontproperties=custom_font)
    
    ax.set_xlabel("Day", fontsize=16, fontproperties=custom_font)
    ax.set_ylabel("Negative Sentiment (0-1)", fontsize=16, fontproperties=custom_font)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(days_str, fontsize=14, fontproperties=custom_font)
    ax.set_yticklabels([f"{tick:.2f}" for tick in ax.get_yticks()], fontsize=14, fontproperties=custom_font)

    # Clean up plot appearance
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.legend(fontsize=14, loc='upper right', prop=custom_font)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)
    
    # Update cache
    cache["image_data"] = buffer
    cache["last_updated"] = datetime.datetime.now()
    
    logging.info("Visualization updated successfully")
    return buffer

# this should render the graph and checks if we should refresh it
async def generate_graph():
    # checks if the current time passes that 1 AM update condition
    if should_update() or cache["image_data"] is None:
        # logs that we’re regenerating, then calls the function that 
        # does all the data fetching and forecasting (formerly process_data())
        logging.info("Regenerating forecast and graph.")
        buf = generate_forecast_and_plot()
        if not buf:
            fallback = io.BytesIO()
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Forecast Unavailable", ha='center', va='center', fontsize=20)
            plt.savefig(fallback, format='png')
            fallback.seek(0)
            return fallback
        return buf
    else:
        logging.info("Serving cached graph.")
        cache["image_data"].seek(0)
        return cache["image_data"]

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Sentiment Forecast API is running!",
        "last_updated": cache["last_updated"].isoformat() if cache["last_updated"] else None
    }

@app.get("/graph.png")
async def get_graph():
    try:
        img = await generate_graph()
        return Response(content=img.getvalue(), media_type="image/png")
    except Exception as e:
        logging.error(f"Error generating graph: {e}")
        return Response(content="Error generating graph", status_code=500)
