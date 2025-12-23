"""
Integration tests for the video deletion API endpoint.

These tests verify the complete end-to-end deletion flow including
API endpoint behavior, error handling, and status code correctness.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from fastapi import HTTPException

from database.deletion_service import DeletionResult, VerificationResult, VideoDeletionService
from database.r2_connector import R2DeletionResult
from database.pinecone_connector import PineconeDeletionResult

# Use anyio for async tests
pytestmark = pytest.mark.anyio


class TestDeletionEndpointSuccess:
    """Test successful deletion scenarios."""

    @pytest.fixture
    def mock_deletion_service(self):
        """Create a mock deletion service."""
        return AsyncMock(spec=VideoDeletionService)

    async def test_successful_deletion_returns_proper_structure(self, mock_deletion_service):
        """Test successful deletion returns proper response structure."""
        # Setup successful deletion result
        mock_deletion_service.delete_video.return_value = DeletionResult(
            success=True,
            hashed_identifier="test_identifier",
            namespace="web-demo",
            r2_result=R2DeletionResult(
                success=True,
                bucket="test-bucket",
                key="test-key",
                file_existed=True,
                bytes_deleted=1024
            ),
            pinecone_result=PineconeDeletionResult(
                success=True,
                chunks_found=3,
                chunks_deleted=3,
                chunk_ids=["chunk1", "chunk2", "chunk3"],
                namespace="web-demo"
            ),
            verification_result=VerificationResult(
                r2_verified=True,
                pinecone_verified=True,
                verification_errors=[],
                timestamp=datetime.utcnow()
            ),
            timestamp=datetime.utcnow()
        )
        
        # Import and patch the endpoint function directly
        from main import Server
        
        # Create server instance and mock its deletion service
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call the internal delete_video method directly (bypassing Modal decorators)
        result = await server._delete_video_internal("test_identifier", "web-demo")
        
        # Verify response structure
        assert result["success"] is True
        assert result["hashed_identifier"] == "test_identifier"
        assert result["namespace"] == "web-demo"
        assert result["message"] == "Video deleted successfully"
        
        # Verify details structure
        assert "details" in result
        assert result["details"]["r2"]["success"] is True
        assert result["details"]["r2"]["file_existed"] is True
        assert result["details"]["r2"]["bytes_deleted"] == 1024
        assert result["details"]["pinecone"]["success"] is True
        assert result["details"]["pinecone"]["chunks_found"] == 3
        assert result["details"]["pinecone"]["chunks_deleted"] == 3
        
        # Verify verification details
        assert "verification" in result
        assert result["verification"]["r2_verified"] is True
        assert result["verification"]["pinecone_verified"] is True
        assert result["verification"]["errors"] == []

    async def test_successful_deletion_with_missing_r2_data(self, mock_deletion_service):
        """Test successful deletion when R2 data doesn't exist."""
        # Setup deletion result with missing R2 data
        mock_deletion_service.delete_video.return_value = DeletionResult(
            success=True,
            hashed_identifier="test_identifier",
            namespace="web-demo",
            r2_result=R2DeletionResult(
                success=True,
                bucket="test-bucket",
                key="test-key",
                file_existed=False,  # File didn't exist
                bytes_deleted=0
            ),
            pinecone_result=PineconeDeletionResult(
                success=True,
                chunks_found=2,
                chunks_deleted=2,
                chunk_ids=["chunk1", "chunk2"],
                namespace="web-demo"
            ),
            timestamp=datetime.utcnow()
        )
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call endpoint
        result = await server._delete_video_internal("test_identifier", "web-demo")
        
        # Should still be successful
        assert result["success"] is True
        assert result["details"]["r2"]["file_existed"] is False
        assert result["details"]["pinecone"]["chunks_deleted"] == 2

    async def test_successful_deletion_with_missing_pinecone_data(self, mock_deletion_service):
        """Test successful deletion when Pinecone data doesn't exist."""
        # Setup deletion result with missing Pinecone data
        mock_deletion_service.delete_video.return_value = DeletionResult(
            success=True,
            hashed_identifier="test_identifier",
            namespace="web-demo",
            r2_result=R2DeletionResult(
                success=True,
                bucket="test-bucket",
                key="test-key",
                file_existed=True,
                bytes_deleted=2048
            ),
            pinecone_result=PineconeDeletionResult(
                success=True,
                chunks_found=0,  # No chunks found
                chunks_deleted=0,
                chunk_ids=[],
                namespace="web-demo"
            ),
            timestamp=datetime.utcnow()
        )
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call endpoint
        result = await server._delete_video_internal("test_identifier", "web-demo")
        
        # Should still be successful
        assert result["success"] is True
        assert result["details"]["r2"]["file_existed"] is True
        assert result["details"]["pinecone"]["chunks_found"] == 0


class TestDeletionEndpointErrors:
    """Test error scenarios and status codes."""

    @pytest.fixture
    def mock_deletion_service(self):
        """Create a mock deletion service."""
        return AsyncMock(spec=VideoDeletionService)

    async def test_empty_identifier_returns_400(self, mock_deletion_service):
        """Test empty hashed_identifier returns 400 Bad Request."""
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Test with empty string
        with pytest.raises(HTTPException) as exc_info:
            await server._delete_video_internal("", "web-demo")
        
        # Should be HTTPException with 400 status
        assert exc_info.value.status_code == 400
        assert "ValidationError" in str(exc_info.value.detail)

    async def test_whitespace_identifier_returns_400(self, mock_deletion_service):
        """Test whitespace-only hashed_identifier returns 400 Bad Request."""
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Test with whitespace
        with pytest.raises(HTTPException) as exc_info:
            await server._delete_video_internal("   ", "web-demo")
        
        # Should be HTTPException with 400 status
        assert exc_info.value.status_code == 400
        assert "ValidationError" in str(exc_info.value.detail)

    async def test_production_environment_returns_403(self, mock_deletion_service):
        """Test production environment returns 403 Forbidden."""
        # Setup deletion service to return environment restriction error
        mock_deletion_service.delete_video.return_value = DeletionResult(
            success=False,
            hashed_identifier="test_identifier",
            namespace="web-demo",
            r2_result=R2DeletionResult(
                success=False,
                bucket="",
                key="",
                file_existed=False,
                error_message="Not attempted due to environment restriction"
            ),
            pinecone_result=PineconeDeletionResult(
                success=False,
                chunks_found=0,
                chunks_deleted=0,
                chunk_ids=[],
                namespace="web-demo",
                error_message="Not attempted due to environment restriction"
            ),
            error_message="Deletion not allowed in prod environment"
        )
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call endpoint
        with pytest.raises(HTTPException) as exc_info:
            await server._delete_video_internal("test_identifier", "web-demo")
        
        # Should be HTTPException with 403 status
        assert exc_info.value.status_code == 403
        assert "AuthorizationError" in str(exc_info.value.detail)

    async def test_invalid_identifier_format_returns_400(self, mock_deletion_service):
        """Test invalid identifier format returns 400 Bad Request."""
        # Setup deletion service to return validation error
        mock_deletion_service.delete_video.return_value = DeletionResult(
            success=False,
            hashed_identifier="invalid_identifier",
            namespace="web-demo",
            r2_result=R2DeletionResult(
                success=False,
                bucket="",
                key="",
                file_existed=False,
                error_message="Invalid identifier format"
            ),
            pinecone_result=PineconeDeletionResult(
                success=False,
                chunks_found=0,
                chunks_deleted=0,
                chunk_ids=[],
                namespace="web-demo",
                error_message="Invalid identifier format"
            ),
            error_message="Invalid identifier: Base64 decode error"
        )
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call endpoint
        with pytest.raises(HTTPException) as exc_info:
            await server._delete_video_internal("invalid_identifier", "web-demo")
        
        # Should be HTTPException with 400 status
        assert exc_info.value.status_code == 400
        assert "ValidationError" in str(exc_info.value.detail)

    async def test_video_not_found_returns_404(self, mock_deletion_service):
        """Test video not found in either system returns 404 Not Found."""
        # Setup deletion result where video doesn't exist in either system
        mock_deletion_service.delete_video.return_value = DeletionResult(
            success=True,  # Service considers this successful (nothing to delete)
            hashed_identifier="nonexistent_identifier",
            namespace="web-demo",
            r2_result=R2DeletionResult(
                success=True,
                bucket="test-bucket",
                key="test-key",
                file_existed=False,  # File didn't exist
                bytes_deleted=0
            ),
            pinecone_result=PineconeDeletionResult(
                success=True,
                chunks_found=0,  # No chunks found
                chunks_deleted=0,
                chunk_ids=[],
                namespace="web-demo"
            ),
            timestamp=datetime.utcnow()
        )
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call endpoint
        with pytest.raises(HTTPException) as exc_info:
            await server._delete_video_internal("nonexistent_identifier", "web-demo")
        
        # Should be HTTPException with 404 status
        assert exc_info.value.status_code == 404
        assert "NotFoundError" in str(exc_info.value.detail)

    async def test_r2_failure_returns_500(self, mock_deletion_service):
        """Test R2 deletion failure returns 500 Internal Server Error."""
        # Setup deletion result with R2 failure
        mock_deletion_service.delete_video.return_value = DeletionResult(
            success=False,
            hashed_identifier="test_identifier",
            namespace="web-demo",
            r2_result=R2DeletionResult(
                success=False,
                bucket="test-bucket",
                key="test-key",
                file_existed=True,
                error_message="Network timeout during R2 deletion"
            ),
            pinecone_result=PineconeDeletionResult(
                success=True,
                chunks_found=2,
                chunks_deleted=2,
                chunk_ids=["chunk1", "chunk2"],
                namespace="web-demo"
            ),
            error_message="Deletion incomplete or failed verification"
        )
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call endpoint
        with pytest.raises(HTTPException) as exc_info:
            await server._delete_video_internal("test_identifier", "web-demo")
        
        # Should be HTTPException with 500 status
        assert exc_info.value.status_code == 500
        assert "StorageError" in str(exc_info.value.detail)

    async def test_pinecone_failure_returns_500(self, mock_deletion_service):
        """Test Pinecone deletion failure returns 500 Internal Server Error."""
        # Setup deletion result with Pinecone failure
        mock_deletion_service.delete_video.return_value = DeletionResult(
            success=False,
            hashed_identifier="test_identifier",
            namespace="web-demo",
            r2_result=R2DeletionResult(
                success=True,
                bucket="test-bucket",
                key="test-key",
                file_existed=True,
                bytes_deleted=1024
            ),
            pinecone_result=PineconeDeletionResult(
                success=False,
                chunks_found=3,
                chunks_deleted=0,
                chunk_ids=[],
                namespace="web-demo",
                error_message="Pinecone connection timeout"
            ),
            error_message="Deletion incomplete or failed verification"
        )
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call endpoint
        with pytest.raises(HTTPException) as exc_info:
            await server._delete_video_internal("test_identifier", "web-demo")
        
        # Should be HTTPException with 500 status
        assert exc_info.value.status_code == 500
        assert "StorageError" in str(exc_info.value.detail)

    async def test_unexpected_exception_returns_500(self, mock_deletion_service):
        """Test unexpected exception returns 500 Internal Server Error."""
        # Setup deletion service to raise unexpected exception
        mock_deletion_service.delete_video.side_effect = RuntimeError("Unexpected database error")
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call endpoint
        with pytest.raises(HTTPException) as exc_info:
            await server._delete_video_internal("test_identifier", "web-demo")
        
        # Should be HTTPException with 500 status
        assert exc_info.value.status_code == 500
        assert "InternalError" in str(exc_info.value.detail)


class TestDeletionEndpointParameters:
    """Test parameter handling and validation."""

    @pytest.fixture
    def mock_deletion_service(self):
        """Create a mock deletion service."""
        return AsyncMock(spec=VideoDeletionService)

    async def test_default_namespace_parameter(self, mock_deletion_service):
        """Test default namespace parameter is used correctly."""
        # Setup successful deletion result
        mock_deletion_service.delete_video.return_value = DeletionResult(
            success=True,
            hashed_identifier="test_identifier",
            namespace="web-demo",  # Default namespace
            r2_result=R2DeletionResult(
                success=True,
                bucket="test-bucket",
                key="test-key",
                file_existed=True,
                bytes_deleted=1024
            ),
            pinecone_result=PineconeDeletionResult(
                success=True,
                chunks_found=1,
                chunks_deleted=1,
                chunk_ids=["chunk1"],
                namespace="web-demo"
            ),
            timestamp=datetime.utcnow()
        )
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call endpoint without namespace parameter
        result = await server._delete_video_internal("test_identifier")
        
        # Verify deletion service was called with default namespace
        mock_deletion_service.delete_video.assert_called_once_with("test_identifier", "web-demo")
        
        # Verify response uses default namespace
        assert result["namespace"] == "web-demo"

    async def test_custom_namespace_parameter(self, mock_deletion_service):
        """Test custom namespace parameter is used correctly."""
        # Setup successful deletion result
        mock_deletion_service.delete_video.return_value = DeletionResult(
            success=True,
            hashed_identifier="test_identifier",
            namespace="custom-namespace",
            r2_result=R2DeletionResult(
                success=True,
                bucket="test-bucket",
                key="test-key",
                file_existed=True,
                bytes_deleted=1024
            ),
            pinecone_result=PineconeDeletionResult(
                success=True,
                chunks_found=1,
                chunks_deleted=1,
                chunk_ids=["chunk1"],
                namespace="custom-namespace"
            ),
            timestamp=datetime.utcnow()
        )
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call endpoint with custom namespace
        result = await server._delete_video_internal("test_identifier", "custom-namespace")
        
        # Verify deletion service was called with custom namespace
        mock_deletion_service.delete_video.assert_called_once_with("test_identifier", "custom-namespace")
        
        # Verify response uses custom namespace
        assert result["namespace"] == "custom-namespace"


class TestDeletionServiceIntegration:
    """Test integration with the deletion service."""

    async def test_deletion_service_called_with_correct_parameters(self):
        """Test that the deletion service is called with correct parameters."""
        mock_deletion_service = AsyncMock(spec=VideoDeletionService)
        mock_deletion_service.delete_video.return_value = DeletionResult(
            success=True,
            hashed_identifier="test_id",
            namespace="test_ns",
            r2_result=R2DeletionResult(
                success=True,
                bucket="bucket",
                key="key",
                file_existed=True,
                bytes_deleted=100
            ),
            pinecone_result=PineconeDeletionResult(
                success=True,
                chunks_found=1,
                chunks_deleted=1,
                chunk_ids=["chunk1"],
                namespace="test_ns"
            ),
            timestamp=datetime.utcnow()
        )
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        # Call with specific parameters
        await server._delete_video_internal("test_id", "test_ns")
        
        # Verify service was called correctly
        mock_deletion_service.delete_video.assert_called_once_with("test_id", "test_ns")

    async def test_error_handling_preserves_service_errors(self):
        """Test that errors from the deletion service are properly handled."""
        mock_deletion_service = AsyncMock(spec=VideoDeletionService)
        
        # Test different error scenarios
        error_scenarios = [
            # Environment restriction
            DeletionResult(
                success=False,
                hashed_identifier="test_id",
                namespace="web-demo",
                r2_result=R2DeletionResult(success=False, bucket="", key="", file_existed=False, error_message="Environment restriction"),
                pinecone_result=PineconeDeletionResult(success=False, chunks_found=0, chunks_deleted=0, chunk_ids=[], namespace="web-demo", error_message="Environment restriction"),
                error_message="Deletion not allowed in prod environment"
            ),
            # Validation error
            DeletionResult(
                success=False,
                hashed_identifier="invalid_id",
                namespace="web-demo",
                r2_result=R2DeletionResult(success=False, bucket="", key="", file_existed=False, error_message="Invalid format"),
                pinecone_result=PineconeDeletionResult(success=False, chunks_found=0, chunks_deleted=0, chunk_ids=[], namespace="web-demo", error_message="Invalid format"),
                error_message="Invalid identifier: decode error"
            )
        ]
        
        from main import Server
        server = Server()
        server.deletion_service = mock_deletion_service
        
        for error_result in error_scenarios:
            mock_deletion_service.delete_video.return_value = error_result
            
            with pytest.raises(HTTPException):
                await server._delete_video_internal("test_id", "web-demo")