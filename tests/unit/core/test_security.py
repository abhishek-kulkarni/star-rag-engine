from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from app.core.security import get_current_user


@pytest.mark.asyncio
async def test_get_current_user_valid_token():
    """Verify that a valid Firebase token returns the UID."""
    mock_credentials = MagicMock(spec=HTTPAuthorizationCredentials)
    mock_credentials.credentials = "valid_token"

    mock_decoded_token = {"uid": "user_123"}

    with patch(
        "app.core.security.auth.verify_id_token", return_value=mock_decoded_token
    ) as mock_verify:
        uid = await get_current_user(mock_credentials)

        assert uid == "user_123"
        mock_verify.assert_called_once_with("valid_token")


@pytest.mark.asyncio
async def test_get_current_user_missing_uid():
    """Verify that a token missing the UID raises 401."""
    mock_credentials = MagicMock(spec=HTTPAuthorizationCredentials)
    mock_credentials.credentials = "token_no_uid"

    mock_decoded_token = {"email": "test@example.com"}  # No uid

    with patch(
        "app.core.security.auth.verify_id_token", return_value=mock_decoded_token
    ):
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(mock_credentials)

        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "missing uid" in excinfo.value.detail


@pytest.mark.asyncio
async def test_get_current_user_invalid_token():
    """Verify that an invalid or expired token raises 401."""
    mock_credentials = MagicMock(spec=HTTPAuthorizationCredentials)
    mock_credentials.credentials = "invalid_token"

    with patch(
        "app.core.security.auth.verify_id_token", side_effect=Exception("Token expired")
    ):
        with pytest.raises(HTTPException) as excinfo:
            await get_current_user(mock_credentials)

        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Token expired" in excinfo.value.detail
