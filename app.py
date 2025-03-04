import os
import re
import datetime
import time
import io
import joblib
import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from fastapi import FastAPI
from starlette.responses import Response

# ------------------ Configuration ------------------ #
# Google Drive file URL (ensure this link points to a valid PyTorch checkpoint)
MODEL_URL = "https://drive.google.com/uc?id=1mzeWB1SeTrYLchnUSMXO8pGM4lJGc0md"
MODEL_PATH = "score_predictor.pth"

# Custom font path (ensure the file is in your working directory)
FONT_PATH = "AfacadFlux-VariableFont_slnt,wght[1].ttf"

# ------------------ Model Download ------------------ #
def download_model():
    """Download and validate the model from Google Drive if needed."""
    if os.path.exists(MODEL_PATH):
        try:
            _ = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
            print("✅ Model is already downloaded and valid.")
            return
        except Exception:
            print("⚠️ Model file appears corrupt. Re-downloading...")

    print("⬇️ Downloading model from Google Drive...")
    response = requests.get(MODEL_URL, stream=True)
    if "text/html" in response.headers.get("Content-Type", ""):
        raise Exception("❌ Download failed: Received HTML content. Check your Google Drive link.")
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("✅ Download complete. Validating model...")
    try:
        _ = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        print("✅ Model validated successfully!")
    except Exception as e:
        os.remove(MODEL_PATH)  # Remove invalid file
        raise Exception(f"❌ Downloaded file is not a valid PyTorch model: {e}")

download_model()

# ------------------ Initialize FastAPI ------------------ #
app = FastAPI()

# ------------------ Load Pre-trained Assets ------------------ #
try:
    autovectorizer = joblib.load('AutoVectorizer.pkl')
    autoclassifier = joblib.load('AutoClassifier.pkl')
    sentiment_model = joblib.load('sentiment_forecast_model.pkl')
    print("✅ Pre-trained vectorizer & classifier loaded successfully!")
except Exception as e:
    print(f"❌ Error loading vectorizer/classifier: {e}")

# ------------------ Load Transformer Tokenizer ------------------ #
# (This is used solely to obtain the vocab size for your ScorePredictor model.)
from transformers import AutoTokenizer
TRANSFORMER_MODEL = "cardiffnlp/xlm-twitter-politics-sentiment"
try:
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    vocab_size = tokenizer.vocab_size
    print(f"✅ Tokenizer loaded. Vocab size: {vocab_size}")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    vocab_size = 250002  # fallback value

# ------------------ Define ScorePredictor Model ------------------ #
class ScorePredictor(nn.Module):
    def __init__(self, vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1):
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

# ------------------ Load the PyTorch Model ------------------ #
try:
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    print(f"✅ Model checkpoint keys: {checkpoint.keys()}")
    score_model = ScorePredictor()
    score_model.load_state_dict(checkpoint)
    score_model.eval()
    print("✅ Score model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading score model: {e}")

# ------------------ Define Predict Function ------------------ #
def predict_score(text: str) -> float:
    """Tokenize the text and predict sentiment score using the PyTorch model."""
    if not text:
        return 0.0
    # Preprocess text: remove URLs and unwanted characters
    text_clean = re.sub(r'http\S+', '', text).strip().lower()
    tokens = text_clean.split()  # simple whitespace tokenization
    # Convert tokens to token IDs; here we use a very simple mapping (for demo purposes)
    # In practice, use tokenizer.encode(...) if your model expects proper IDs.
    token_ids = [min(hash(token) % vocab_size, vocab_size - 1) for token in tokens]
    # Pad/truncate to fixed length (e.g., 20 tokens)
    max_len = 20
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        score = score_model(input_tensor)[0].item()
    return score

# ------------------ Reddit Data Processing ------------------ #
import praw
# (Make sure you have set your Reddit API credentials as environment variables)
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent='Cipher/0.0.1',
    check_for_async=False
)

subreddits = ["florida", "kratom", "ohio", "libertarian", "walkaway", "truechristian", "jordanpeterson"]
def fetch_all_recent_posts(subreddit_name, start_time, limit=100):
    """Fetch posts from a subreddit that are newer than start_time."""
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
        print(f"❌ Error fetching posts from r/{subreddit_name}: {e}")
    return posts

# ------------------ Build DataFrame & Forecast ------------------ #
def compute_forecast():
    """Fetch Reddit posts, compute daily sentiment, and forecast future sentiment."""
    start_date = datetime.datetime.utcnow() - datetime.timedelta(days=14)
    all_posts = []
    for sub in subreddits:
        posts = fetch_all_recent_posts(sub, start_date, limit=100)
        all_posts.extend(posts)
        time.sleep(1)  # avoid rate limits

    # Filter posts using classifier (only keep posts predicted as relevant)
    filtered_posts = []
    for post in all_posts:
        try:
            vector = autovectorizer.transform([post['post_text']])
            if autoclassifier.predict(vector)[0] == 1:
                filtered_posts.append(post)
        except Exception as e:
            print(f"❌ Error classifying post: {e}")
    if not filtered_posts:
        raise Exception("No relevant posts found.")
    df = pd.DataFrame(filtered_posts)
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    df = df.sort_values(by=['date_only'])
    df['sentiment_score'] = df['post_text'].apply(predict_score)

    unique_dates = sorted(df['date_only'].unique(), reverse=True)[:14]
    filtered_df = df[df['date_only'].isin(unique_dates)]
    daily_sentiment = filtered_df.groupby('date_only')['sentiment_score'].median()
    # If fewer than 14 days of data, pad with mean
    if len(daily_sentiment) < 14:
        mean_sentiment = daily_sentiment.mean()
        padding = [mean_sentiment] * (14 - len(daily_sentiment))
        daily_sentiment = np.concatenate([daily_sentiment.values, padding])
        daily_sentiment = pd.Series(daily_sentiment)
    # Use the pre-loaded sentiment_model to predict a 7-day forecast
    sentiment_scores_np = daily_sentiment.values.reshape(1, -1)
    forecast = sentiment_model.predict(sentiment_scores_np)[0]  # assume output is a 7-element vector
    return forecast

# ------------------ Custom Font for Plotting ------------------ #
import matplotlib.font_manager as fm
try:
    custom_font = fm.FontProperties(fname=FONT_PATH)
except Exception as e:
    print(f"⚠️ Could not load custom font from {FONT_PATH}: {e}")
    custom_font = None

# ------------------ FastAPI Endpoints ------------------ #
@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "Cipher Sentiment Forecast API is running!"}

@app.get("/graph.png")
def generate_graph():
    """Fetch data, generate forecast graph, and return it as PNG."""
    try:
        forecast = compute_forecast()  # get 7-day forecast from sentiment_model
    except Exception as e:
        return Response(content=f"❌ Error computing forecast: {e}", media_type="text/plain")

    # Generate date labels for the next 7 days
    today = datetime.date.today()
    days = [today + datetime.timedelta(days=i) for i in range(7)]
    days_str = [day.strftime('%a %m/%d') for day in days]

    # Smooth the forecast curve
    xnew = np.linspace(0, 6, 300)
    spline = make_interp_spline(np.arange(7), forecast, k=3)
    forecast_smooth = spline(xnew)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(xnew, forecast_smooth, color='#244B48', alpha=0.4)
    ax.plot(xnew, forecast_smooth, color='#244B48', lw=3, label='Forecast')
    ax.scatter(np.arange(7), forecast, color='#244B48', s=100, zorder=5)
    title_kwargs = {"fontsize": 22, "fontweight": "bold", "pad": 20}
    label_kwargs = {"fontsize": 16}
    tick_kwargs = {"fontsize": 14}
    if custom_font is not None:
        title_kwargs["fontproperties"] = custom_font
        label_kwargs["fontproperties"] = custom_font
        tick_kwargs["fontproperties"] = custom_font

    ax.set_title("7-Day Political Sentiment Forecast", **title_kwargs)
    ax.set_xlabel("Day", **label_kwargs)
    ax.set_ylabel("Negative Sentiment (0-1)", **label_kwargs)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(days_str, **tick_kwargs)
    # Format y-axis tick labels with custom font if possible
    ax.set_yticklabels([f"{tick:.2f}" for tick in ax.get_yticks()], **tick_kwargs)

    ax.legend(fontsize=14, loc="upper right", prop=custom_font if custom_font is not None else None)
    # Optionally remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    # Save figure to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)  # close figure to free memory
    return Response(content=buf.getvalue(), media_type="image/png")
