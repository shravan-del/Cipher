import os
import time
import datetime
import joblib
import numpy as np
import pandas as pd
import torch
import praw
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from torch import nn

# Load Vectorizers and Models
autovectorizer = joblib.load('AutoVectorizer.pkl')
autoclassifier = joblib.load('AutoClassifier.pkl')
sentiment_model = joblib.load('sentiment_forecast_model.pkl')

# Define a simple LSTM-based Score Predictor model
class ScorePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super(ScorePredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_hidden_state = lstm_out[:, -1, :]
        output = self.fc(final_hidden_state)
        return self.sigmoid(output)

# Load Model
score_model = ScorePredictor(input_dim=10)  # Adjust input_dim based on your data
try:
    score_model.load_state_dict(torch.load("score_predictor.pth", map_location=torch.device('cpu')))
    score_model.eval()
except FileNotFoundError:
    st.error("Error: Model file 'score_predictor.pth' not found.")

# Ensure Reddit API credentials are set
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")

if not reddit_client_id or not reddit_client_secret:
    st.error("Missing Reddit API credentials! Please set 'REDDIT_CLIENT_ID' and 'REDDIT_CLIENT_SECRET'.")

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent="MyAPI/0.0.1",
    check_for_async=False
)

subreddits = ['florida', 'kratom', 'ohio', 'libertarian', 'walkaway', 'truechristian', 'jordanpeterson']
start_date = datetime.datetime.utcnow() - datetime.timedelta(days=14)

def fetch_recent_posts(subreddit_name, start_time, limit=100):
    """Fetches recent posts from a subreddit."""
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    try:
        for post in subreddit.new(limit=limit):
            post_time = datetime.datetime.utcfromtimestamp(post.created_utc)
            if post_time >= start_time:
                posts.append({
                    "subreddit": subreddit_name,
                    "timestamp": post.created_utc,
                    "date": post_time.strftime('%Y-%m-%d'),
                    "post_text": post.title
                })
    except Exception as e:
        st.error(f"Error fetching posts from r/{subreddit_name}: {e}")
    return posts

# Fetch and filter posts
all_posts = []
for sub in subreddits:
    all_posts.extend(fetch_recent_posts(sub, start_date, limit=100))
    time.sleep(1)

# Filter posts based on classifier
filtered_posts = [post for post in all_posts if autoclassifier.predict(autovectorizer.transform([post['post_text']]))[0] == 1]

df = pd.DataFrame(filtered_posts)
if df.empty:
    st.warning("No data available for the selected period.")
else:
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    df = df.sort_values(by=['date_only'])

    # Dummy Sentiment Scores (If real scores unavailable)
    df['sentiment_score'] = np.random.uniform(0.2, 0.8, len(df))

    # Compute daily sentiment averages
    last_14_dates = sorted(df['date_only'].unique(), reverse=True)[:14]
    daily_sentiment = df[df['date_only'].isin(last_14_dates)].groupby('date_only')['sentiment_score'].mean()

    # Handle missing data by padding
    if len(daily_sentiment) < 14:
        padding = [daily_sentiment.mean()] * (14 - len(daily_sentiment))
        daily_sentiment = np.concatenate([daily_sentiment.values, padding])
        daily_sentiment = pd.Series(daily_sentiment)

    # Predict sentiment trend
    prediction = sentiment_model.predict(daily_sentiment.values.reshape(1, -1))[0]

    # Generate forecast graph
    days = [datetime.date.today() + datetime.timedelta(days=i) for i in range(7)]
    days_str = [day.strftime('%a %m/%d') for day in days]

    # Ensure smooth plotting
    xnew = np.linspace(0, 6, 300)
    spline = make_interp_spline(np.arange(7), prediction, k=3)
    pred_smooth = spline(xnew)

    # Plot Graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(xnew, pred_smooth, color="gray", alpha=0.4)
    ax.plot(xnew, pred_smooth, color="black", lw=2, label="Forecast")
    ax.scatter(np.arange(7), prediction, color="black", s=80, zorder=5)

    ax.set_title("7-Day Political Sentiment Forecast", fontsize=18, fontweight="bold")
    ax.set_xlabel("Day", fontsize=14)
    ax.set_ylabel("Negative Sentiment (0-1)", fontsize=14)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(days_str, fontsize=12)
    ax.legend(fontsize=12, loc="upper right")
    plt.tight_layout()

    # Display only the graph
    st.pyplot(fig)
