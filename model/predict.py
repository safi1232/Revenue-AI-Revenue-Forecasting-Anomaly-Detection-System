# ============================================================
# model/predict.py
# REVENUE PREDICTION + ANOMALY DETECTION
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "revenue_prediction_model.pkl"
)


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

model_package = joblib.load(MODEL_PATH)


# ============================================================
# EXTRACT INFORMATION
# ============================================================

# IMPORTANT:
# The pickle contains a dictionary.
# The actual XGBoost model is inside ["model"].

model = model_package["model"]

features = model_package["features"]

target = model_package.get(
    "target",
    "Revenue"
)

ANOMALY_THRESHOLD = model_package.get(
    "anomaly_threshold",
    20.0
)

MODEL_VERSION = "1.0.0"


# ============================================================
# DEBUG INFORMATION
# ============================================================

print("=" * 60)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 60)

print(
    "Model:",
    type(model)
)

print(
    "Number of features:",
    len(features)
)

print(
    "Target:",
    target
)

print(
    "Anomaly threshold:",
    ANOMALY_THRESHOLD
)

print("=" * 60)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_output(user_data: dict):

    try:

        # ----------------------------------------------------
        # Convert input dictionary to DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [user_data]
        )


        # ----------------------------------------------------
        # Make sure all training features exist
        # ----------------------------------------------------

        for feature in features:

            if feature not in input_df.columns:

                input_df[feature] = np.nan


        # ----------------------------------------------------
        # Keep EXACT training feature order
        # ----------------------------------------------------

        input_df = input_df[
            features
        ]


        # ----------------------------------------------------
        # Convert numeric values
        # ----------------------------------------------------

        for column in input_df.columns:

            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="coerce"
            )


        # ----------------------------------------------------
        # Fill missing values
        # ----------------------------------------------------

        train_medians = model_package.get(
            "train_medians",
            {}
        )


        for column in input_df.columns:

            if column in train_medians:

                input_df[column] = (
                    input_df[column]
                    .fillna(
                        train_medians[column]
                    )
                )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )


        prediction = float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )


        # ----------------------------------------------------
        # HANDLE log1p MODEL
        # ----------------------------------------------------

        target_transform = model_package.get(
            "target_transform",
            ""
        )


        if target_transform.lower() in [
            "log1p",
            "log"
        ]:

            predicted_revenue = (
                np.expm1(
                    prediction
                )
            )

        else:

            predicted_revenue = prediction


        predicted_revenue = max(
            0.0,
            float(
                predicted_revenue
            )
        )


        # ----------------------------------------------------
        # ANOMALY SCORE
        # ----------------------------------------------------
        #
        # For a future revenue forecast, there is no actual
        # revenue yet. Therefore we return the prediction as
        # NORMAL by default.
        #
        # Anomaly detection should be performed when actual
        # revenue becomes available.
        # ----------------------------------------------------

        anomaly_score = 0.0

        is_anomaly = False

        anomaly_status = "NORMAL"


        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {

            "predicted_revenue":
                predicted_revenue,

            "is_anomaly":
                is_anomaly,

            "anomaly_status":
                anomaly_status,

            "anomaly_score":
                anomaly_score,

            "model_version":
                MODEL_VERSION
        }


    except Exception as e:

        raise RuntimeError(
            f"Prediction failed: {str(e)}"
        )