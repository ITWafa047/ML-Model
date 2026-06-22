from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from core.config import API_KEY


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to check for API key in the request headers."""
    async def dispatch(self, request: Request, call_next):

        if request.url.path in ["/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)

        if request.url.path.startswith("/public"):
            return await call_next(request)

        api_key = request.headers.get("x-api-key")

        if api_key != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"}
            )

        return await call_next(request)