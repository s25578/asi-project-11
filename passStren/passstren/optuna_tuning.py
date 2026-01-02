import pandas as pd
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')


def run_optuna(data_path: str = "data/password_data.csv", sample: float = 0.1, n_trials: int = 20):
    print("Loading data...")
    df = pd.read_csv(data_path, on_bad_lines='skip').dropna()
    print(f"Loaded {len(df)} samples")
    
    if sample and 0 < sample < 1:
        df = df.sample(frac=sample, random_state=42)
        print(f"Using {len(df)} samples")
    
    print("Vectorizing passwords...")
    vectorizer = TfidfVectorizer(analyzer='char')
    X = vectorizer.fit_transform(df["password"])
    y = df["strength"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'random_state': 42
        }
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        return scores.mean()
    
    print(f"Optimizing hyperparameters with {n_trials} trials...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest accuracy: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    print("Training final model...")
    best_model = RandomForestClassifier(**study.best_params)
    best_model.fit(X_train, y_train)
    
    print(f"Test accuracy: {best_model.score(X_test, y_test):.4f}")
    
    return best_model, study


if __name__ == "__main__":
    run_optuna()
