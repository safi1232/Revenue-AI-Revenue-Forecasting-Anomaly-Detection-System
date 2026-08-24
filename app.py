from fastapi import FastAPI, HTTPException

from model.predict import (
    predict_output,
    model,
    MODEL_VERSION
)

from schemas.respons_prediction import PredictionResponse
from schemas.user_input import UserInput


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Revenue Prediction & Anomaly Detection API",

    description=(
        "API for predicting revenue and detecting "
        "revenue anomalies."
    ),

    version=MODEL_VERSION
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message":
            "Revenue Prediction & Anomaly Detection API",

        "status":
            "ok",

        "version":
            MODEL_VERSION
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status":
            "ok",

        "version":
            MODEL_VERSION,

        "model_loaded":
            model is not None
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(user_input: UserInput):

    try:

        # Convert Pydantic object to dictionary

        user_data = (
            user_input.model_dump()
        )

        # Make prediction

        prediction = predict_output(
            user_data
        )

        # Return response

        return PredictionResponse(

            predicted_revenue=
                prediction[
                    "predicted_revenue"
                ],

            is_anomaly=
                prediction[
                    "is_anomaly"
                ],

            anomaly_status=
                prediction[
                    "anomaly_status"
                ],

            anomaly_score=
                prediction[
                    "anomaly_score"
                ],

            model_version=
                prediction[
                    "model_version"
                ]
        )

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )