import os
import datetime
import time
import joblib
import numpy as np
import pandas as pd
import torch
import praw
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from torch import nn
from fastapi import FastAPI
from starlette.responses import Response
import io

# Initialize FastAPI
app = FastAPI()

# Load pre-trained vectorizer and classifier
autovectorizer = joblib.load('AutoVectorizer.pkl')
autoclassifier = joblib.load('AutoClassifier.pkl')
sentiment_model = joblib.load('sentiment_forecast_model.pkl')

# Load the saved model checkpoint and extract vocab size
checkpoint = torch.load("score_predictor.pth", map_location=torch.device('cpu'))
vocab_size = checkpoint["embedding.weight"].shape[0]  # Extract vocab size from model

# Define ScorePredictor with the correct parameters
class ScorePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1):
        super(ScorePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        final_hidden_state = lstm_out[:, -1, :]
        output = self.fc(final_hidden_state)
        return self.sigmoid(output)

# Load the model with the correct vocab size
score_model = ScorePredictor(vocab_size=vocab_size)
score_model.load_state_dict(checkpoint)
score_model.eval()

# Set up Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent='MyAPI/0.0.1',
    check_for_async=False
)

subreddits = ['florida', 'kratom', 'ohio', 'libertarian', 'walkaway', 'truechristian', 'jordanpeterson']
start_date = datetime.datetime.utcnow() - datetime.timedelta(days=14)

def fetch_all_recent_posts(subreddit_name, start_time, limit=100):
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
        print(f"Error fetching posts from r/{subreddit_name}: {e}")
    return posts

def predict_score(text):
    if not text:
        return 0.0
    tokenized_input = torch.randint(0, vocab_size, (1, 20))  # Simulating tokenized input
    with torch.no_grad():
        score = score_model(tokenized_input, None)[0].item()
    return score

# Fetch Reddit data
all_posts = []
for sub in subreddits:
    all_posts.extend(fetch_all_recent_posts(sub, start_date, limit=100))
    time.sleep(1)

filtered_posts = [post for post in all_posts if autoclassifier.predict(autovectorizer.transform([post['post_text']]))[0] == 1]

df = pd.DataFrame(filtered_posts)

@app.get("/graph.png")
def generate_graph():
    if df.empty:
        return Response(content="No data available.", media_type="text/plain")

    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    df = df.sort_values(by=['date_only'])
    df['sentiment_score'] = df['post_text'].apply(predict_score)
    
    last_14_dates = sorted(df['date_only'].unique(), reverse=True)[:14]
    daily_sentiment = df[df['date_only'].isin(last_14_dates)].groupby('date_only')['sentiment_score'].mean()
    
    if len(daily_sentiment) < 14:
        padding = [daily_sentiment.mean()] * (14 - len(daily_sentiment))
        daily_sentiment = np.concatenate([daily_sentiment.values, padding])
        daily_sentiment = pd.Series(daily_sentiment)
    
    prediction = sentiment_model.predict(daily_sentiment.values.reshape(1, -1))[0]

    # Plot graph
    days = [datetime.date.today() + datetime.timedelta(days=i) for i in range(7)]
    days_str = [day.strftime('%a %m/%d') for day in days]
    xnew = np.linspace(0, 6, 300)
    spline = make_interp_spline(np.arange(7), prediction, k=3)
    pred_smooth = spline(xnew)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(xnew, pred_smooth, color='#244B48', alpha=0.4)
    ax.plot(xnew, pred_smooth, color='#244B48', lw=3, label='Forecast')
    ax.scatter(np.arange(7), prediction, color='#244B48', s=100, zorder=5)
    ax.set_title("7-Day Sentiment Forecast", fontsize=18, fontweight='bold')
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Negative Sentiment", fontsize=12)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(days_str, fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return Response(content=img.getvalue(), media_type="image/png")
