import pandas as pd
from pycaret.classification import ClassificationExperiment


def run_pycaret(data_path: str = "data/password_data.csv", sample: float = 0.1):
    print("Loading data...")
    df = pd.read_csv(data_path, on_bad_lines='skip').dropna()
    print(f"Loaded {len(df)} samples")
    
    if sample and 0 < sample < 1:
        df = df.sample(frac=sample, random_state=42)
        print(f"Using {len(df)} samples")
    
    df["strength"] = df["strength"].map({0: "Weak", 1: "Medium", 2: "Strong"})
    
    print("Setting up PyCaret...")
    clf = ClassificationExperiment()
    clf.setup(data=df, target='strength', session_id=42, verbose=False, 
              text_features=['password'], log_experiment=False, html=False)
    
    print("Comparing models...")
    best_model = clf.compare_models(n_select=1, turbo=True, sort='Accuracy', verbose=True)
    
    print("Generating predictions...")
    predictions = clf.predict_model(best_model)
    
    scoring = clf.pull()
    
    print(f"\nBest Model: {type(best_model).__name__}")
    print("\nScoring:")
    print(scoring)
    print("\nSample Predictions:")
    print(predictions[['password', 'strength', 'prediction_label']].head(10))
    
    print("\nSaving model...")
    clf.save_model(best_model, 'model')
    print("Model saved as model.pkl")
    
    return best_model, predictions, scoring


def predict_password(password: str):
    print("Loading model...")
    clf = ClassificationExperiment()
    model = clf.load_model('model')
    
    df = pd.DataFrame({'password': [password], 'strength': ['Unknown']})
    result = clf.predict_model(model, data=df)
    
    prediction = result['prediction_label'].iloc[0]
    
    print(f"\nPassword: {password}")
    print(f"Predicted Strength: {prediction}")


if __name__ == "__main__":
    run_pycaret()
