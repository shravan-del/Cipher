import os
import time
import datetime
import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from torch import nn

# Load vectorizer and classifier
autovectorizer = joblib.load('AutoVectorizer.pkl')
autoclassifier = joblib.load('AutoClassifier.pkl')
sentiment_model = joblib.load('sentiment_forecast_model.pkl')

# Fix PyTorch Model Loading
class ScorePredictor(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256, output_dim=1):
        super(ScorePredictor, self).__init__()
        self.embedding = nn.Linear(embedding_dim, hidden_dim)  # Fix incorrect embedding layer
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

# Simulated Data for Sentiment Forecast (Replace this with actual processing)
np.random.seed(42)
prediction = np.random.uniform(0.3, 0.7, 7)  # Simulated 7-day sentiment scores

# Plot Sentiment Forecast
days = [datetime.date.today() + datetime.timedelta(days=i) for i in range(7)]
days_str = [day.strftime('%a %m/%d') for day in days]
xnew = np.linspace(0, 6, 300)
spline = make_interp_spline(np.arange(7), prediction, k=3)
pred_smooth = spline(xnew)

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

# Display only the Graph
st.pyplot(fig)