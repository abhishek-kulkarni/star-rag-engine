from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, UploadFile

from app.api.v1.endpoints.documents import upload_document


@pytest.mark.asyncio
async def test_upload_document_missing_filename_unit():
    """
    Manually invokes the endpoint to reach the filename validation line.
    This bypasses FastAPI's 422 validation to hit the 400 check in our code.
    """
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = None  # Force the missing filename state

    mock_db = MagicMock()

    with pytest.raises(HTTPException) as excinfo:
        await upload_document(file=mock_file, current_user="test_user", db=mock_db)

    assert excinfo.value.status_code == 400
    assert "filename" in excinfo.value.detail.lower()
