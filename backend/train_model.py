import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
df = pd.read_csv("data/IMDB Dataset.csv")  # path relative to backend folder

# Encode labels: positive=1, negative=0
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

X = df["review"]
y = df["label"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "sentiment_model.pkl")
print("Model trained and saved as sentiment_model.pkl")
