from pydantic import BaseModel, Field
from typing import Annotated


class UserInput(BaseModel):

    Date: Annotated[
        str,
        Field(
            ...,
            description="Prediction date in YYYY-MM-DD format",
            examples=["2026-08-24"]
        )
    ]

    Orders: Annotated[
        int,
        Field(
            ...,
            ge=0,
            description="Number of orders",
            examples=[250]
        )
    ]

    Customers: Annotated[
        int,
        Field(
            ...,
            ge=0,
            description="Number of unique customers",
            examples=[180]
        )
    ]

    Products: Annotated[
        int,
        Field(
            ...,
            ge=0,
            description="Number of unique products sold",
            examples=[95]
        )
    ]