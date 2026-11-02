from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class APIError(Exception):
    def __init__(
        self,
        status_code: int = 400,
        message: str = "bad request",
        type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.type = type
        self.param = param
        self.code = code


def error_body(
    message: str,
    type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict:
    return {
        "error": {
            "message": message,
            "type": type,
            "param": param,
            "code": code,
        }
    }


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(exc.message, exc.type, exc.param, exc.code),
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=error_body(str(exc.errors()[:1]), type="invalid_request_error"),
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
