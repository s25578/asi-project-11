from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import pandas as pd
import joblib
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
app = FastAPI()

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


@app.get("/models")
async def list_models():
    models = []
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith(".joblib"):
                models.append(f.replace(".joblib", ""))
    return {"models": models}


def do_training(df, new_name, old_path=None):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if old_path and os.path.exists(old_path):
        model = joblib.load(old_path)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X, y)

    preds = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, average='macro', zero_division=0)),
        "recall": float(recall_score(y, preds, average='macro', zero_division=0))
    }

    save_as = os.path.join(MODEL_DIR, f"{new_name}.joblib")
    joblib.dump(model, save_as)
    return metrics


@app.post("/continue-train")
async def continue_train(
        model_name: str = Form(...),
        new_model_name: str = Form(...),
        train_input: UploadFile = File(...)
):
    if not train_input.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="only csv files")

    tmp = f"tmp_{train_input.filename}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(train_input.file, f)

    df = pd.read_csv(tmp)
    os.remove(tmp)

    if df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="need features + target column")

    old = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(old):
        old = None

    metrics = do_training(df, new_model_name, old)

    return {"metrics": metrics, "new_model": new_model_name}


@app.post("/predict")
async def predict(
        model_name: str = Form(...),
        input: UploadFile = File(...)
):
    if not model_name.strip():
        raise HTTPException(status_code=400, detail="model name required")

    if not input.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="csv only")

    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(path):
        available = [f.replace(".joblib", "") for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]
        raise HTTPException(status_code=404, detail=f"no such model. available: {available}")

    tmp = f"pred_{input.filename}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(input.file, f)

    data = pd.read_csv(tmp)
    os.remove(tmp)

    model = joblib.load(path)
    predictions = model.predict(data).tolist()

    return {"predictions": predictions}