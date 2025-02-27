import os
import datetime
import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from torch import nn
from fastapi import FastAPI
from starlette.responses import Response
import io
import requests

# ✅ Google Drive Model URL & Path
MODEL_URL = "https://drive.google.com/uc?id=1mzeWB1SeTrYLchnUSMXO8pGM4lJGc0md"
MODEL_PATH = "score_predictor.pth"

def download_model():
    """Download and validate the model from Google Drive if needed."""
    if os.path.exists(MODEL_PATH):
        try:
            _ = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
            print("✅ Model is already downloaded and valid.")
            return
        except Exception:
            print("⚠️ Model file is corrupt. Re-downloading...")

    print("⬇️ Downloading model from Google Drive...")
    response = requests.get(MODEL_URL, stream=True)

    if "text/html" in response.headers.get("Content-Type", ""):
        raise Exception("❌ Download failed: Received an invalid file. Check the Google Drive link.")

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    print("✅ Download complete. Validating model...")

    try:
        _ = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        print("✅ Model validated successfully!")
    except Exception as e:
        os.remove(MODEL_PATH)  # Delete invalid file
        raise Exception(f"❌ Downloaded file is not a valid PyTorch model: {e}")

# ✅ Ensure model is downloaded before loading
download_model()

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Load pre-trained vectorizer and classifier
try:
    autovectorizer = joblib.load('AutoVectorizer.pkl')
    autoclassifier = joblib.load('AutoClassifier.pkl')
    sentiment_model = joblib.load('sentiment_forecast_model.pkl')
    print("✅ Pre-trained vectorizer & classifier loaded successfully!")
except Exception as e:
    print(f"❌ Error loading vectorizer/classifier: {e}")

# ✅ **Define Model Architecture**
class ScorePredictor(nn.Module):
    def __init__(self, vocab_size=250002, embedding_dim=128, hidden_dim=256, output_dim=1):
        super(ScorePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        final_hidden_state = lstm_out[:, -1, :]
        output = self.fc(final_hidden_state)
        return self.sigmoid(output)

# ✅ **Load PyTorch Model Correctly**
try:
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    print(f"✅ Model checkpoint keys: {checkpoint.keys()}")

    score_model = ScorePredictor()
    score_model.load_state_dict(checkpoint)
    score_model.eval()
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model: {e}")

# ✅ **Simulated Data for Sentiment Forecast**
np.random.seed(42)
prediction = np.random.uniform(0.3, 0.7, 7)  # Simulated 7-day sentiment scores

@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "Sentiment Forecast API is running!"}

@app.get("/graph.png")
def generate_graph():
    """Generate and return a sentiment forecast graph."""
    try:
        # ✅ Generate X-axis labels
        days = [datetime.date.today() + datetime.timedelta(days=i) for i in range(7)]
        days_str = [day.strftime('%a %m/%d') for day in days]

        # ✅ Smooth the curve
        xnew = np.linspace(0, 6, 300)
        spline = make_interp_spline(np.arange(7), prediction, k=3)
        pred_smooth = spline(xnew)

        # ✅ Create the plot
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

        # ✅ Save to in-memory image file
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        return Response(content=img.getvalue(), media_type="image/png")

    except Exception as e:
        return {"error": f"❌ Failed to generate graph: {e}"}