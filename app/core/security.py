from typing import Annotated

import firebase_admin
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth

# Initialize Firebase Admin SDK
# Note: In production, ensure GOOGLE_APPLICATION_CREDENTIALS is set
# or use credentials.Certificate() for local development.
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app()

security = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    """
    FastAPI dependency that validates the Firebase JWT and returns the user's UID.
    This UID serves as the partition key for all multi-tenant operations.
    """
    token = credentials.credentials
    try:
        # Decode and verify the Firebase ID Token
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token.get("uid")

        if not uid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing uid",
            )

        return uid

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
        ) from e
