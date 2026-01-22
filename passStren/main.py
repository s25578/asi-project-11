import contextlib
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ml_models = {}

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Instead of training, we just load the files
    print("🚀 Loading pre-trained model...")
    try:
        ml_models["vectorizer"] = joblib.load("vectorizer.joblib")
        ml_models["classifier"] = joblib.load("password_model.joblib")
        print("✅ API Ready!")
    except FileNotFoundError:
        print("❌ Error: Model files not found. Run train_model.py first.")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)


# Define the input data format
class PasswordRequest(BaseModel):
    password: str


# Define the output data format
class PasswordResponse(BaseModel):
    password: str
    strength: str


@app.get("/")
def read_root():
    return {"status": "Password Strength API is running"}


@app.post("/predict", response_model=PasswordResponse)
def predict_strength(request: PasswordRequest):
    # Check if model is loaded
    if "classifier" not in ml_models or "vectorizer" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    # 1. Vectorize the input password
    # Note: We wrap request.password in a list [] because transform expects an iterable
    vectorized_password = ml_models["vectorizer"].transform([request.password])

    # 2. Predict
    prediction = ml_models["classifier"].predict(vectorized_password)

    # 3. Return result
    return {
        "password": request.password,
        "strength": prediction[0]
    }