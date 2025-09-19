
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# Enable frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("sentiment_model.pkl")

class TextData(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TextData):
    text = [data.text]  
    proba = model.predict_proba(text)[0]  
    positive_prob = proba[1]
    
    # Threshold for neutral
    if 0.45 < positive_prob < 0.55:
        sentiment = "neutral"
    elif positive_prob >= 0.55:
        sentiment = "positive"
    else:
        sentiment = "negative"
    
    return {"sentiment": sentiment}
