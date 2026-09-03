# code/adapter/response/response_custom.py

from pydantic import BaseModel
from fastapi.responses import JSONResponse


class ResponseModel(BaseModel):
    messages: str
    status: int

    class Config:
        extra = "allow"


def HandleSuccess(message: str, **kwargs):
    response_data = {
        "messages": message,
        "status": 200,
        **kwargs,
    }
    return JSONResponse(content=response_data, status_code=200)
