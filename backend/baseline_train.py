from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load IMDB dataset
dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

# Vectorize text
vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train = vec.fit_transform(train_texts)
X_test = vec.transform(test_texts)

# Train Logistic Regression
clf = LogisticRegression(max_iter=1000, solver="saga")
clf.fit(X_train, train_labels)

# Evaluate
preds = clf.predict(X_test)
print(classification_report(test_labels, preds))

# Save models
joblib.dump(vec, "models/tfidf_vec.joblib")
joblib.dump(clf, "models/logreg_model.joblib")
print(" Model saved in backend/models/")
