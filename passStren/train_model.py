import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def train():
    print("⏳ Loading data...")
    data = pd.read_csv("data/password_data.csv", on_bad_lines='skip').dropna()
    data["strength"] = data["strength"].map({0: "Weak", 1: "Medium", 2: "Strong"}).dropna()

    print("⏳ Vectorizing...")
    vectorizer = TfidfVectorizer(analyzer='char')
    X = vectorizer.fit_transform(data["password"])
    y = data["strength"]

    print("⏳ Training model (this is the slow part)...")
    clf = RandomForestClassifier(n_jobs=-1) # n_jobs=-1 uses all your CPU cores to speed it up
    clf.fit(X, y)

    print("💾 Saving files...")
    joblib.dump(clf, "password_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    print("✅ Done! You now have 'password_model.joblib' and 'vectorizer.joblib'.")

if __name__ == "__main__":
    train()