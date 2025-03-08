# -*- coding: utf-8 -*-
import os
import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import asyncpraw
import asyncio
import gradio as gr

# Global settings
num_days = 14
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
MODEL_PATH = "cardiffnlp/xlm-twitter-politics-sentiment"



# Minimal model class definition (required)
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

# Load models
sentiment_model = joblib.load('sentiment_forecast_model.pkl')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
score_model = ScorePredictor(tokenizer.vocab_size)
score_model.load_state_dict(torch.load("score_predictor.pth", map_location=torch.device('cpu')))
score_model.eval()
print("Models loaded successfully.")

# Function to fetch posts from Reddit
async def get_posts(subreddit_name, time_filter='month'):

    # Initialize asyncpraw Reddit client
    async_reddit = asyncpraw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent="sentimentForecastAgent"
)




    
    subreddit = await async_reddit.subreddit(subreddit_name)
    posts = []

    async for post in subreddit.top(time_filter=time_filter, limit=25):
        posts.append({
            "date": datetime.datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            "post_text": post.title
        })

    return posts

# Function to calculate sentiment
def calculate_sentiment(text):
    if not text:
        return 0.0
    else:
        encoded = tokenizer(text.split(), return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            score_val = score_model(encoded["input_ids"], encoded["attention_mask"])[0].item()
        return score_val

# Function to generate sentiment forecast
async def generate_forecast(subreddit, num_days=14):
    # Fetch posts asynchronously
    posts = await get_posts(subreddit)

    # Create DataFrame and process dates
    df = pd.DataFrame(posts)
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    df = df.sort_values('date_only')

    # Calculate sentiment scores
    df['sentiment_score'] = df['post_text'].apply(calculate_sentiment)

    # Create a complete date index for the last num_days and group by date
    full_dates = sorted([datetime.date.today() - datetime.timedelta(days=i) for i in range(num_days)])
    daily = df.groupby('date_only')['sentiment_score'].mean().reindex(full_dates, fill_value=0.0)
    historical = daily.values.tolist()

    # Forecast using the pre-loaded sentiment_model
    forecast = sentiment_model.predict(np.array(historical).reshape(1, -1))[0]

    # Create forecast plot
    today = datetime.date.today()
    forecast_dates = [today + datetime.timedelta(days=i) for i in range(7)]
    x = np.arange(7)
    xnew = np.linspace(0, 6, 300)
    spline = make_interp_spline(x, forecast, k=min(3, len(forecast)-1))
    smooth = spline(xnew)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.fill_between(xnew, smooth, color='#244B48', alpha=0.4)
    ax.plot(xnew, smooth, color='#244B48', lw=3, label='Forecast')
    ax.scatter(x, forecast, color='#244B48', s=100)
    ax.set_title("7-Day Political Sentiment Forecast", fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Negative Sentiment (0-1)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime('%a %m/%d') for d in forecast_dates], fontsize=12, rotation=45)
    ax.legend(fontsize=14, loc='upper right')
    plt.tight_layout()

    if not posts:
        summary = "Subreddit not found or criteria not met for Cypher."
    else:
        summary = f"r/{subreddit} has loaded!"

    return fig, summary

# Gradio interface
async def run_forecast(subreddit):
    fig, summary = await generate_forecast(subreddit)
    return fig, summary

# Gradio app
with gr.Blocks(title="Political Sentiment Forecast") as demo:
    gr.Markdown("# Reddit Political Sentiment Forecast")
    gr.Markdown("Analyze recent Reddit posts to forecast political sentiment for the next 7 days.")
    
    subreddit_input = gr.Textbox(label="Subreddit (without r/)", placeholder="e.g. politics", value="politics")
    output_text = gr.Textbox(label="Summary", lines=2)
    submit_btn = gr.Button("Generate Forecast")
    output_plot = gr.Plot(label="Forecast Plot")
    
    submit_btn.click(fn=run_forecast, inputs=subreddit_input, outputs=[output_plot, output_text])

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
