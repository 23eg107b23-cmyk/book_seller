import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1) Load dataset
df = pd.read_csv("data/Top-100 Trending Books.csv")
print("Columns:", df.columns)

# 2) Sort by rating (highest first)
df = df.sort_values(by="rating", ascending=False).reset_index(drop=True)

# 3) Create target label (Top 50 → Bestseller = 1, Otherwise = 0)
df["is_bestseller"] = df["Rank"].apply(lambda x: 1 if x <= 50 else 0)

# 4) Use book title + genre + author as text features
df["text"] = df["book title"] + " " + df["genre"] + " " + df["author"]

X = df["text"]
y = df["is_bestseller"]

# 5) Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6) Vectorize text
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7) Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 8) Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# 9) Save model and vectorizer
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("Training completed. Files saved: model.joblib & vectorizer.joblib ✅")
