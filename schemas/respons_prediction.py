from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):

    predicted_revenue: float = Field(
        ...,
        description="Predicted revenue"
    )

    is_anomaly: bool = Field(
        ...,
        description="Whether the predicted revenue is anomalous"
    )

    anomaly_status: str = Field(
        ...,
        description="Normal or Anomaly"
    )

    anomaly_score: float = Field(
        ...,
        description="Anomaly score"
    )

    model_version: str = Field(
        ...,
        description="Model version"
    )