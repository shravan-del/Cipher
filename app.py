import os
import re
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
from transformers import AutoTokenizer
from torch import nn



autovectorizer = joblib.load('AutoVectorizer.pkl')
autoclassifier = joblib.load('AutoClassifier.pkl')
sentiment_model = joblib.load('sentiment_forecast_model.pkl')

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
score_model.load_state_dict(torch.load("score_predictor.pth", map_location=torch.device('cpu')))
score_model.eval()

reddit = praw.Reddit(
    client_id = os.getenv("REDDIT_CLIENT_ID"),
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
        st.error(f"Error fetching posts from r/{subreddit_name}: {e}")
    return posts

def predict_score(text):
    if not text:
        return 0.0
    encoded_input = tokenizer(text.split(), return_tensors='pt', padding=True, truncation=True)
    input_ids, attention_mask = encoded_input["input_ids"], encoded_input["attention_mask"]
    with torch.no_grad():
        score = score_model(input_ids, attention_mask)[0].item()
    return score

all_posts = []
for sub in subreddits:
    all_posts.extend(fetch_all_recent_posts(sub, start_date, limit=100))
    time.sleep(1)

filtered_posts = [post for post in all_posts if autoclassifier.predict(autovectorizer.transform([post['post_text']]))[0] == 1]

df = pd.DataFrame(filtered_posts)
if df.empty:
    st.warning("No data available for the selected period.")
else:
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
    
    # Plot
    days = [datetime.date.today() + datetime.timedelta(days=i) for i in range(7)]
    days_str = [day.strftime('%a %m/%d') for day in days]
    xnew = np.linspace(0, 6, 300)
    spline = make_interp_spline(np.arange(7), prediction, k=3)
    pred_smooth = spline(xnew)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(xnew, pred_smooth, color='#244B48', alpha=0.4)
    ax.plot(xnew, pred_smooth, color='#244B48', lw=3, label='Forecast')
    ax.scatter(np.arange(7), prediction, color='#244B48', s=100, zorder=5)
    ax.set_title("7-Day Political Sentiment Forecast", fontsize=22, fontweight='bold')
    ax.set_xlabel("Day", fontsize=16)
    ax.set_ylabel("Negative Sentiment (0-1)", fontsize=16)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(days_str, fontsize=14)
    ax.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    
    st.pyplot(fig)



