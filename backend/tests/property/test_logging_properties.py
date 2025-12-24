"""
Property-based tests for video deletion logging functionality.

These tests use Hypothesis to generate random test data and verify that
logging properties hold across all valid inputs. Each property test
runs a minimum of 100 iterations to ensure comprehensive coverage.
"""

import asyncio
import logging
from hypothesis import given, strategies as st, settings
from unittest.mock import MagicMock
from io import StringIO

from database.deletion_service import VideoDeletionService
from database.r2_connector import R2Connector, R2DeletionResult
from database.pinecone_connector import PineconeConnector, PineconeDeletionResult


class TestLoggingProperties:
    """Property-based tests for deletion service logging operations."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Create a string buffer to capture log output
        self.log_buffer = StringIO()
        self.log_handler = logging.StreamHandler(self.log_buffer)
        self.log_handler.setLevel(logging.DEBUG)
        
        # Get the logger and add our handler
        self.logger = logging.getLogger('database.deletion_service')
        self.logger.addHandler(self.log_handler)
        self.logger.setLevel(logging.DEBUG)
        
        # Store original handlers to restore later
        self.original_handlers = self.logger.handlers[:-1]  # All except our test handler

    def teardown_method(self):
        """Clean up after each test method."""
        # Remove our test handler and restore original handlers
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()

    def get_log_output(self) -> str:
        """Get the captured log output as a string."""
        return self.log_buffer.getvalue()

    def clear_log_output(self):
        """Clear the captured log output."""
        self.log_buffer.seek(0)
        self.log_buffer.truncate(0)

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    )
    @settings(max_examples=100)
    def test_unauthorized_attempt_logging_property(self, hashed_identifier, namespace):
        """
        Feature: video-deletion, Property 10: Unauthorized Attempt Logging
        
        For any deletion attempt in prod environment, the system should create 
        a security log entry with request details.
        
        Validates: Requirements 3.4, 7.4
        """
        # Create mock connectors
        mock_r2 = MagicMock(spec=R2Connector)
        mock_pinecone = MagicMock(spec=PineconeConnector)
        
        # Create deletion service in prod environment
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="prod"
        )
        
        self.clear_log_output()
        
        # Attempt deletion (should be blocked) - use asyncio.run to handle async
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Unauthorized attempt should be logged
        log_output = self.get_log_output()
        
        # Verify security logging occurred
        assert "Unauthorized deletion attempt" in log_output
        assert "prod environment" in log_output
        assert hashed_identifier in log_output
        
        # Verify the deletion was blocked
        assert result.success is False
        assert "not allowed" in result.error_message
        
        # Verify no actual deletion operations were attempted
        mock_r2.delete_video_file.assert_not_called()
        mock_pinecone.delete_by_metadata.assert_not_called()

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    )
    @settings(max_examples=100)
    def test_request_logging_property(self, hashed_identifier, namespace):
        """
        Feature: video-deletion, Property 16: Request Logging
        
        For any deletion request received, the system should create a log entry 
        containing the hashed_identifier and namespace.
        
        Validates: Requirements 7.1
        """
        # Create mock connectors with successful responses
        mock_r2 = MagicMock(spec=R2Connector)
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=True, bucket="test-bucket", key="test-key", 
            file_existed=True, bytes_deleted=1024
        )
        mock_r2.verify_deletion.return_value = True
        mock_r2._decode_path.return_value = ("bucket", "namespace", "filename")
        
        mock_pinecone = MagicMock(spec=PineconeConnector)
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=True, chunks_found=2, chunks_deleted=2, 
            chunk_ids=["chunk1", "chunk2"], namespace=namespace
        )
        mock_pinecone.find_chunks_by_video.return_value = []
        
        # Create deletion service in dev environment
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        self.clear_log_output()
        
        # Make deletion request
        asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Request should be logged with identifier and namespace
        log_output = self.get_log_output()
        
        assert "Video deletion request" in log_output
        assert f"identifier={hashed_identifier}" in log_output
        assert f"namespace={namespace}" in log_output

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        r2_chunks_deleted=st.integers(min_value=0, max_value=10000),
        pinecone_chunks_deleted=st.integers(min_value=0, max_value=100)
    )
    @settings(max_examples=100)
    def test_result_logging_property(self, hashed_identifier, namespace, r2_chunks_deleted, pinecone_chunks_deleted):
        """
        Feature: video-deletion, Property 17: Result Logging
        
        For any completed deletion operation, the system should log the results 
        including the number of items removed from each storage system.
        
        Validates: Requirements 7.2
        """
        # Create mock connectors with parameterized responses
        mock_r2 = MagicMock(spec=R2Connector)
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=True, bucket="test-bucket", key="test-key", 
            file_existed=r2_chunks_deleted > 0, bytes_deleted=r2_chunks_deleted
        )
        mock_r2.verify_deletion.return_value = True
        mock_r2._decode_path.return_value = ("bucket", "namespace", "filename")
        
        mock_pinecone = MagicMock(spec=PineconeConnector)
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=True, chunks_found=pinecone_chunks_deleted, 
            chunks_deleted=pinecone_chunks_deleted, 
            chunk_ids=[f"chunk{i}" for i in range(pinecone_chunks_deleted)], 
            namespace=namespace
        )
        mock_pinecone.find_chunks_by_video.return_value = []
        
        # Create deletion service in dev environment
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        self.clear_log_output()
        
        # Perform deletion
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Results should be logged with items removed from each system
        log_output = self.get_log_output()
        
        if r2_chunks_deleted > 0:
            assert "R2 deletion successful" in log_output
            assert f"{r2_chunks_deleted} bytes" in log_output
        else:
            assert "R2 deletion completed: file" in log_output and "did not exist" in log_output
            
        assert f"Pinecone deletion successful: {pinecone_chunks_deleted} chunks deleted" in log_output
        
        # Verify final completion logging
        if result.success:
            assert "Video deletion completed successfully" in log_output
        else:
            assert "Video deletion failed or incomplete" in log_output

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        error_message=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Zs')))
    )
    @settings(max_examples=100)
    def test_error_logging_property(self, hashed_identifier, namespace, error_message):
        """
        Feature: video-deletion, Property 18: Error Logging
        
        For any error during deletion, the system should log detailed error 
        information including error type, message, and stack trace.
        
        Validates: Requirements 7.3
        """
        # Create mock connectors with error responses
        mock_r2 = MagicMock(spec=R2Connector)
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=False, bucket="test-bucket", key="test-key", 
            file_existed=False, error_message=error_message
        )
        mock_r2._decode_path.return_value = ("bucket", "namespace", "filename")
        
        mock_pinecone = MagicMock(spec=PineconeConnector)
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=False, chunks_found=0, chunks_deleted=0, 
            chunk_ids=[], namespace=namespace, error_message=error_message
        )
        
        # Create deletion service in dev environment
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        self.clear_log_output()
        
        # Perform deletion (should encounter errors)
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Errors should be logged with detailed information
        log_output = self.get_log_output()
        
        # Verify error logging occurred - should have either specific error logs OR final failure log
        assert ("R2 deletion failed" in log_output or 
                "Pinecone deletion failed" in log_output or
                "Video deletion failed or incomplete" in log_output), f"Expected error logging not found in: {log_output}"
        
        # Verify error message appears in logs
        assert error_message in log_output, f"Error message '{error_message}' not found in logs"
        
        # Verify the operation failed
        assert result.success is False

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        r2_verified=st.booleans(),
        pinecone_verified=st.booleans()
    )
    @settings(max_examples=100)
    def test_verification_logging_property(self, hashed_identifier, namespace, r2_verified, pinecone_verified):
        """
        Feature: video-deletion, Property 19: Verification Logging
        
        For any deletion verification performed, the system should log the 
        verification results including success/failure status.
        
        Validates: Requirements 7.5
        """
        # Create mock connectors with successful deletion but variable verification
        mock_r2 = MagicMock(spec=R2Connector)
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=True, bucket="test-bucket", key="test-key", 
            file_existed=True, bytes_deleted=1024
        )
        mock_r2.verify_deletion.return_value = r2_verified
        mock_r2._decode_path.return_value = ("bucket", "namespace", "filename")
        
        mock_pinecone = MagicMock(spec=PineconeConnector)
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=True, chunks_found=2, chunks_deleted=2, 
            chunk_ids=["chunk1", "chunk2"], namespace=namespace
        )
        
        # Mock verification results
        if pinecone_verified:
            mock_pinecone.find_chunks_by_video.return_value = []  # No remaining chunks
        else:
            mock_pinecone.find_chunks_by_video.return_value = ["remaining_chunk"]  # Some chunks remain
        
        # Create deletion service in dev environment
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        self.clear_log_output()
        
        # Perform deletion
        asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Verification should be logged with results
        log_output = self.get_log_output()
        
        # Verify verification logging occurred
        assert "Starting deletion verification" in log_output
        
        if r2_verified and pinecone_verified:
            assert "Deletion verification successful: both systems confirmed" in log_output
        else:
            assert "Deletion verification issues" in log_output
            
            if not r2_verified:
                assert "R2 verification failed" in log_output
            if not pinecone_verified:
                assert "Pinecone verification failed" in log_output

    @given(
        invalid_identifier=st.text(max_size=10, alphabet=st.characters(blacklist_categories=('Lu', 'Ll', 'Nd'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    )
    @settings(max_examples=100)
    def test_validation_error_logging_property(self, invalid_identifier, namespace):
        """
        Property: Validation errors should be logged with detailed information.
        
        For any deletion request with invalid identifier format,
        the system should log the validation error with details.
        """
        # Create mock connectors
        mock_r2 = MagicMock(spec=R2Connector)
        mock_r2._decode_path.side_effect = ValueError("Invalid identifier format")
        
        mock_pinecone = MagicMock(spec=PineconeConnector)
        
        # Create deletion service in dev environment
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        self.clear_log_output()
        
        # Attempt deletion with invalid identifier
        result = asyncio.run(service.delete_video(invalid_identifier, namespace))
        
        # Property: Validation error should be logged
        log_output = self.get_log_output()
        
        assert "Failed to decode identifier" in log_output
        assert invalid_identifier in log_output
        assert "Invalid identifier" in log_output
        
        # Verify the operation failed
        assert result.success is False
        assert "Invalid identifier" in result.error_message

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    )
    @settings(max_examples=100)
    def test_environment_validation_logging_property(self, hashed_identifier, namespace):
        """
        Property: Environment validation should be logged for all requests.
        
        For any deletion request, the system should log the environment 
        validation result regardless of environment type.
        """
        # Test both dev and prod environments
        for environment in ["dev", "prod"]:
            # Create mock connectors
            mock_r2 = MagicMock(spec=R2Connector)
            mock_pinecone = MagicMock(spec=PineconeConnector)
            
            # Create deletion service
            service = VideoDeletionService(
                r2_connector=mock_r2,
                pinecone_connector=mock_pinecone,
                environment=environment
            )
            
            self.clear_log_output()
            
            # Attempt deletion
            asyncio.run(service.delete_video(hashed_identifier, namespace))
            
            # Property: Environment validation should be logged
            log_output = self.get_log_output()
            
            assert "Environment validation" in log_output
            assert environment in log_output
            assert "deletion_allowed=" in log_output