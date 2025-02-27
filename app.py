import os
import datetime
import time
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from torch import nn
from fastapi import FastAPI
from starlette.responses import Response
import io
from transformers import AutoTokenizer


# Initialize FastAPI
app = FastAPI()

# Load pre-trained vectorizer and classifier
autovectorizer = joblib.load('AutoVectorizer.pkl')
autoclassifier = joblib.load('AutoClassifier.pkl')
sentiment_model = joblib.load('sentiment_forecast_model.pkl')

# Load PyTorch Model
checkpoint = torch.load("score_predictor.pth", map_location=torch.device('cpu'))

# Define ScorePredictor model
class ScorePredictor(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, output_size=1):
        super(ScorePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        lstm_out, _ = self.lstm(input_tensor)
        final_hidden_state = lstm_out[:, -1, :]
        output = self.fc(final_hidden_state)
        return self.sigmoid(output)

# Load the model
score_model = ScorePredictor(input_size=128, hidden_size=256, output_size=1)
score_model.load_state_dict(checkpoint)
score_model.eval()

# Simulated Data for Sentiment Forecast (Replace this with actual Reddit processing)
np.random.seed(42)
prediction = np.random.uniform(0.3, 0.7, 7)  # Simulated 7-day sentiment scores

@app.get("/")
def home():
    return {"message": "Sentiment Forecast API is running!"}

@app.get("/graph.png")
def generate_graph():
    # Generate X-axis labels
    days = [datetime.date.today() + datetime.timedelta(days=i) for i in range(7)]
    days_str = [day.strftime('%a %m/%d') for day in days]

    # Smooth the curve
    xnew = np.linspace(0, 6, 300)
    spline = make_interp_spline(np.arange(7), prediction, k=3)
    pred_smooth = spline(xnew)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(xnew, pred_smooth, color='#244B48', alpha=0.4)
    ax.plot(xnew, pred_smooth, color='#244B48', lw=3, label='Forecast')
    ax.scatter(np.arange(7), prediction, color='#244B48', s=100, zorder=5)
    ax.set_title("7-Day Political Sentiment Forecast", fontsize=16, fontweight='bold')
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Negative Sentiment (0-1)", fontsize=12)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(days_str, fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()

    # Save to in-memory image file
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    return Response(content=img.getvalue(), media_type="image/png")
