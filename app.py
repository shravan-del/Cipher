import os
import time
import datetime
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from fastapi import FastAPI
from fastapi.responses import Response
from io import BytesIO
from torch import nn

app = FastAPI()

# Load ML models
autovectorizer = joblib.load('AutoVectorizer.pkl')
autoclassifier = joblib.load('AutoClassifier.pkl')
sentiment_model = joblib.load('sentiment_forecast_model.pkl')

# Define Model Class
class ScorePredictor(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256, output_dim=1):
        super(ScorePredictor, self).__init__()
        self.embedding = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        embedded = self.embedding(input_tensor)
        lstm_out, _ = self.lstm(embedded)
        final_hidden_state = lstm_out[:, -1, :]
        output = self.fc(final_hidden_state)
        return self.sigmoid(output)

# Load PyTorch Model
score_model = ScorePredictor()
score_model.load_state_dict(torch.load("score_predictor.pth", map_location=torch.device('cpu')))
score_model.eval()

@app.get("/graph.png")
def generate_graph():
    """
    Generate a 7-day sentiment forecast graph and return it as an image.
    """
    np.random.seed(42)
    prediction = np.random.uniform(0.4, 0.6, 7)  # Simulated forecast scores

    days = [datetime.date.today() + datetime.timedelta(days=i) for i in range(7)]
    days_str = [day.strftime('%a %m/%d') for day in days]
    
    xnew = np.linspace(0, 6, 300)
    spline = make_interp_spline(np.arange(7), prediction, k=3)
    pred_smooth = spline(xnew)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(xnew, pred_smooth, color='#AFCBFF', alpha=0.6)
    ax.plot(xnew, pred_smooth, color='#005BBB', lw=3, label='Forecast')
    ax.scatter(np.arange(7), prediction, color='#005BBB', s=100, zorder=5)

    ax.set_title("7-Day Sentiment Forecast", fontsize=16, fontweight='bold')
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Negative Sentiment", fontsize=12)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(days_str, fontsize=10)
    ax.legend(fontsize=10, loc='upper center', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()

    # Save plot to memory buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
