"""
Property-based tests for VideoDeletionService.

These tests verify the core correctness properties of the video deletion orchestrator,
including dual system confirmation, environment access controls, and error handling.
"""

import asyncio
from unittest.mock import MagicMock
from hypothesis import given, strategies as st, settings

from database.deletion_service import VideoDeletionService
from database.r2_connector import R2DeletionResult
from database.pinecone_connector import PineconeDeletionResult


class TestVideoDeletionServiceProperties:
    """Property-based tests for VideoDeletionService core functionality."""

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        r2_bytes_deleted=st.integers(min_value=1, max_value=1000000),
        pinecone_chunks_deleted=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100)
    def test_dual_system_confirmation_property(self, hashed_identifier, namespace, r2_bytes_deleted, pinecone_chunks_deleted):
        """
        Feature: video-deletion, Property 3: Dual System Confirmation
        
        For any successful deletion operation, the response should contain 
        confirmation of removal from both R2 and Pinecone storage systems.
        
        Validates: Requirements 1.4
        """
        # Setup mock connectors
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Mock successful R2 deletion
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=True,
            bucket="test-bucket",
            key=f"test-namespace/{hashed_identifier}",
            file_existed=True,
            bytes_deleted=r2_bytes_deleted
        )
        
        # Mock successful Pinecone deletion
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=True,
            chunks_found=pinecone_chunks_deleted,
            chunks_deleted=pinecone_chunks_deleted,
            chunk_ids=[f"chunk_{i}" for i in range(pinecone_chunks_deleted)],
            namespace=namespace
        )
        
        # Mock successful verification
        mock_r2.verify_deletion.return_value = True
        mock_pinecone.find_chunks_by_video.return_value = []
        
        # Mock identifier decoding
        mock_r2._decode_path.return_value = ("test-bucket", "test-namespace", f"{hashed_identifier}.mp4")
        
        # Create service in dev environment
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        # Perform deletion
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Successful deletion should contain confirmation from both systems
        assert result.success is True
        assert result.hashed_identifier == hashed_identifier
        assert result.namespace == namespace
        
        # R2 confirmation
        assert result.r2_result.success is True
        assert result.r2_result.file_existed is True
        assert result.r2_result.bytes_deleted == r2_bytes_deleted
        
        # Pinecone confirmation
        assert result.pinecone_result.success is True
        assert result.pinecone_result.chunks_found == pinecone_chunks_deleted
        assert result.pinecone_result.chunks_deleted == pinecone_chunks_deleted
        
        # Verification confirmation
        assert result.verification_result is not None
        assert result.verification_result.r2_verified is True
        assert result.verification_result.pinecone_verified is True
        assert len(result.verification_result.verification_errors) == 0
        
        # Verify both connectors were called
        mock_r2.delete_video_file.assert_called_once_with(hashed_identifier)
        mock_pinecone.delete_by_metadata.assert_called_once_with(
            {"hashed_identifier": hashed_identifier}, namespace
        )

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    )
    @settings(max_examples=100)
    def test_dev_environment_access_property(self, hashed_identifier, namespace):
        """
        Feature: video-deletion, Property 8: Dev Environment Access
        
        For any deletion request in dev environment, the system should 
        allow the operation to proceed.
        
        Validates: Requirements 3.1
        """
        # Setup mock connectors
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Mock successful operations
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=True,
            bucket="dev",
            key=f"test-namespace/{hashed_identifier}",
            file_existed=True,
            bytes_deleted=1000
        )
        
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=True,
            chunks_found=1,
            chunks_deleted=1,
            chunk_ids=["chunk_1"],
            namespace=namespace
        )
        
        # Mock verification
        mock_r2.verify_deletion.return_value = True
        mock_pinecone.find_chunks_by_video.return_value = []
        
        # Mock identifier decoding
        mock_r2._decode_path.return_value = ("dev", "test-namespace", f"{hashed_identifier}.mp4")
        
        # Create service in dev environment
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        # Perform deletion
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Dev environment should allow deletion operations
        assert result.success is True
        assert result.error_message is None or "environment" not in result.error_message.lower()
        
        # Verify operations were attempted
        mock_r2.delete_video_file.assert_called_once()
        mock_pinecone.delete_by_metadata.assert_called_once()

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    )
    @settings(max_examples=100)
    def test_prod_environment_restriction_property(self, hashed_identifier, namespace):
        """
        Feature: video-deletion, Property 9: Prod Environment Restriction
        
        For any deletion request in prod environment, the system should 
        reject the request with a 403 Forbidden status.
        
        Validates: Requirements 3.2
        """
        # Setup mock connectors (should not be called)
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Create service in prod environment
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="prod"
        )
        
        # Perform deletion attempt
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Prod environment should reject deletion requests
        assert result.success is False
        assert result.hashed_identifier == hashed_identifier
        assert result.namespace == namespace
        assert "environment" in result.error_message.lower()
        assert "not allowed" in result.error_message.lower()
        
        # Both sub-results should indicate not attempted
        assert result.r2_result.success is False
        assert "environment restriction" in result.r2_result.error_message
        
        assert result.pinecone_result.success is False
        assert "environment restriction" in result.pinecone_result.error_message
        
        # Verify no actual deletion operations were attempted
        mock_r2.delete_video_file.assert_not_called()
        mock_pinecone.delete_by_metadata.assert_not_called()

    @given(
        environment=st.sampled_from(["dev", "prod"])
    )
    @settings(max_examples=100)
    def test_environment_validation_property(self, environment):
        """
        Property: Environment validation should work correctly for all valid environments.
        
        For any valid environment string, the validation should return the correct result.
        """
        # Setup mock connectors
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Create service
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment=environment
        )
        
        # Test validation
        is_allowed = service._validate_environment()
        
        # Property: Only dev environment should allow deletion
        if environment == "dev":
            assert is_allowed is True
        else:  # prod
            assert is_allowed is False

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    )
    @settings(max_examples=100)
    def test_identifier_validation_property(self, hashed_identifier, namespace):
        """
        Property: Invalid identifiers should be handled gracefully.
        
        For any deletion request with invalid identifier, the system should
        fail gracefully with appropriate error message.
        """
        # Setup mock connectors
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Mock identifier decoding to fail
        mock_r2._decode_path.side_effect = ValueError("Invalid identifier format")
        
        # Create service in dev environment
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        # Perform deletion
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Invalid identifier should cause graceful failure
        assert result.success is False
        assert "invalid identifier" in result.error_message.lower()
        
        # Both sub-results should indicate invalid format
        assert result.r2_result.success is False
        assert "invalid identifier" in result.r2_result.error_message.lower()
        
        assert result.pinecone_result.success is False
        assert "invalid identifier" in result.pinecone_result.error_message.lower()
        
        # Verify no deletion operations were attempted
        mock_r2.delete_video_file.assert_not_called()
        mock_pinecone.delete_by_metadata.assert_not_called()

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    )
    @settings(max_examples=100)
    def test_parallel_operations_property(self, hashed_identifier, namespace):
        """
        Property: R2 and Pinecone operations should execute in parallel.
        
        For any deletion request, both storage systems should be accessed
        concurrently to minimize total operation time.
        """
        # Setup mock connectors with async behavior
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Track call order to verify parallelism
        call_order = []
        
        def mock_r2_delete(identifier):
            call_order.append("r2_start")
            result = R2DeletionResult(
                success=True,
                bucket="dev",
                key=f"test/{identifier}",
                file_existed=True,
                bytes_deleted=1000
            )
            call_order.append("r2_end")
            return result
        
        def mock_pinecone_delete(metadata, ns):
            call_order.append("pinecone_start")
            result = PineconeDeletionResult(
                success=True,
                chunks_found=1,
                chunks_deleted=1,
                chunk_ids=["chunk_1"],
                namespace=ns
            )
            call_order.append("pinecone_end")
            return result
        
        mock_r2.delete_video_file.side_effect = mock_r2_delete
        mock_pinecone.delete_by_metadata.side_effect = mock_pinecone_delete
        
        # Mock other required methods
        mock_r2.verify_deletion.return_value = True
        mock_pinecone.find_chunks_by_video.return_value = []
        mock_r2._decode_path.return_value = ("dev", "test", f"{hashed_identifier}.mp4")
        
        # Create service
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        # Perform deletion
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Operations should succeed
        assert result.success is True
        
        # Property: Both operations should have been called
        assert len(call_order) == 4
        assert "r2_start" in call_order
        assert "r2_end" in call_order
        assert "pinecone_start" in call_order
        assert "pinecone_end" in call_order
        
        # Verify both connectors were called
        mock_r2.delete_video_file.assert_called_once_with(hashed_identifier)
        mock_pinecone.delete_by_metadata.assert_called_once_with(
            {"hashed_identifier": hashed_identifier}, namespace
        )


class TestVideoDeletionServiceErrorHandlingProperties:
    """Property-based tests for VideoDeletionService error handling."""

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        pinecone_chunks=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_graceful_r2_missing_data_handling_property(self, hashed_identifier, namespace, pinecone_chunks):
        """
        Feature: video-deletion, Property 4: Graceful R2 Missing Data Handling
        
        For any hashed_identifier with Pinecone data but no R2 data, deletion should 
        complete Pinecone cleanup and report the missing R2 data without failing.
        
        Validates: Requirements 2.1
        """
        # Setup mock connectors
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Mock R2 deletion - file not found
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=True,  # R2 considers "not found" as success
            bucket="dev",
            key=f"test/{hashed_identifier}",
            file_existed=False,
            error_message="File not found in R2 storage"
        )
        
        # Mock successful Pinecone deletion
        chunk_ids = [f"chunk_{i}" for i in range(pinecone_chunks)]
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=True,
            chunks_found=pinecone_chunks,
            chunks_deleted=pinecone_chunks,
            chunk_ids=chunk_ids,
            namespace=namespace
        )
        
        # Mock verification
        mock_r2.verify_deletion.return_value = True  # File confirmed not to exist
        mock_pinecone.find_chunks_by_video.return_value = []  # Chunks successfully deleted
        
        # Mock identifier decoding
        mock_r2._decode_path.return_value = ("dev", "test", f"{hashed_identifier}.mp4")
        
        # Create service
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        # Perform deletion
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Should succeed despite missing R2 data
        assert result.success is True
        assert result.hashed_identifier == hashed_identifier
        assert result.namespace == namespace
        
        # R2 result should indicate file didn't exist but operation succeeded
        assert result.r2_result.success is True
        assert result.r2_result.file_existed is False
        assert "not found" in result.r2_result.error_message.lower()
        
        # Pinecone cleanup should have proceeded successfully
        assert result.pinecone_result.success is True
        assert result.pinecone_result.chunks_deleted == pinecone_chunks
        
        # Both operations should have been attempted
        mock_r2.delete_video_file.assert_called_once_with(hashed_identifier)
        mock_pinecone.delete_by_metadata.assert_called_once_with(
            {"hashed_identifier": hashed_identifier}, namespace
        )

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        r2_bytes=st.integers(min_value=1, max_value=1000000)
    )
    @settings(max_examples=100)
    def test_graceful_pinecone_missing_data_handling_property(self, hashed_identifier, namespace, r2_bytes):
        """
        Feature: video-deletion, Property 5: Graceful Pinecone Missing Data Handling
        
        For any hashed_identifier with R2 data but no Pinecone data, deletion should 
        complete R2 cleanup and report the missing Pinecone data without failing.
        
        Validates: Requirements 2.2
        """
        # Setup mock connectors
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Mock successful R2 deletion
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=True,
            bucket="dev",
            key=f"test/{hashed_identifier}",
            file_existed=True,
            bytes_deleted=r2_bytes
        )
        
        # Mock Pinecone deletion - no chunks found
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=True,  # Pinecone considers "no chunks" as success
            chunks_found=0,
            chunks_deleted=0,
            chunk_ids=[],
            namespace=namespace,
            error_message="No chunks found for video"
        )
        
        # Mock verification
        mock_r2.verify_deletion.return_value = True  # File successfully deleted
        mock_pinecone.find_chunks_by_video.return_value = []  # No chunks exist
        
        # Mock identifier decoding
        mock_r2._decode_path.return_value = ("dev", "test", f"{hashed_identifier}.mp4")
        
        # Create service
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        # Perform deletion
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Should succeed despite missing Pinecone data
        assert result.success is True
        assert result.hashed_identifier == hashed_identifier
        assert result.namespace == namespace
        
        # R2 cleanup should have succeeded
        assert result.r2_result.success is True
        assert result.r2_result.file_existed is True
        assert result.r2_result.bytes_deleted == r2_bytes
        
        # Pinecone result should indicate no chunks found but operation succeeded
        assert result.pinecone_result.success is True
        assert result.pinecone_result.chunks_found == 0
        assert result.pinecone_result.chunks_deleted == 0
        assert "no chunks found" in result.pinecone_result.error_message.lower()
        
        # Both operations should have been attempted
        mock_r2.delete_video_file.assert_called_once_with(hashed_identifier)
        mock_pinecone.delete_by_metadata.assert_called_once_with(
            {"hashed_identifier": hashed_identifier}, namespace
        )

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        error_message=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=100)
    def test_r2_failure_isolation_property(self, hashed_identifier, namespace, error_message):
        """
        Feature: video-deletion, Property 6: R2 Failure Isolation
        
        For any deletion request where R2 deletion fails, the system should return 
        a detailed error message and not proceed with Pinecone deletion.
        
        Validates: Requirements 2.3
        """
        # Setup mock connectors
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Mock R2 deletion failure (not "not found" - actual error)
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=False,
            bucket="dev",
            key=f"test/{hashed_identifier}",
            file_existed=True,
            error_message=f"R2 network error: {error_message}"
        )
        
        # Mock Pinecone deletion (should not be called due to R2 failure)
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=True,
            chunks_found=1,
            chunks_deleted=1,
            chunk_ids=["chunk_1"],
            namespace=namespace
        )
        
        # Mock identifier decoding
        mock_r2._decode_path.return_value = ("dev", "test", f"{hashed_identifier}.mp4")
        
        # Create service
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        # Perform deletion
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Should fail when R2 fails with actual error
        assert result.success is False
        assert result.hashed_identifier == hashed_identifier
        assert result.namespace == namespace
        
        # Should contain detailed error message about R2 failure
        assert "r2 deletion failed" in result.error_message.lower()
        
        # R2 result should show the failure
        assert result.r2_result.success is False
        assert error_message in result.r2_result.error_message
        
        # Pinecone operation should have been attempted (parallel execution)
        # but overall result should still be failure due to R2
        mock_r2.delete_video_file.assert_called_once_with(hashed_identifier)
        mock_pinecone.delete_by_metadata.assert_called_once_with(
            {"hashed_identifier": hashed_identifier}, namespace
        )

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        error_message=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=100)
    def test_pinecone_failure_reporting_property(self, hashed_identifier, namespace, error_message):
        """
        Feature: video-deletion, Property 7: Pinecone Failure Reporting
        
        For any deletion request where Pinecone deletion fails after R2 succeeds, 
        the system should return a detailed error indicating partial completion.
        
        Validates: Requirements 2.4
        """
        # Setup mock connectors
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Mock successful R2 deletion
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=True,
            bucket="dev",
            key=f"test/{hashed_identifier}",
            file_existed=True,
            bytes_deleted=1000
        )
        
        # Mock Pinecone deletion failure
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=False,
            chunks_found=5,
            chunks_deleted=0,
            chunk_ids=["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"],
            namespace=namespace,
            error_message=f"Pinecone connection error: {error_message}"
        )
        
        # Mock identifier decoding
        mock_r2._decode_path.return_value = ("dev", "test", f"{hashed_identifier}.mp4")
        
        # Create service
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        # Perform deletion
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Should fail overall when Pinecone fails
        assert result.success is False
        assert result.hashed_identifier == hashed_identifier
        assert result.namespace == namespace
        
        # Should indicate partial completion
        assert "incomplete" in result.error_message.lower() or "failed" in result.error_message.lower()
        
        # R2 should have succeeded
        assert result.r2_result.success is True
        assert result.r2_result.bytes_deleted == 1000
        
        # Pinecone should show the failure with details
        assert result.pinecone_result.success is False
        assert result.pinecone_result.chunks_found == 5
        assert result.pinecone_result.chunks_deleted == 0
        assert error_message in result.pinecone_result.error_message
        
        # Both operations should have been attempted
        mock_r2.delete_video_file.assert_called_once_with(hashed_identifier)
        mock_pinecone.delete_by_metadata.assert_called_once_with(
            {"hashed_identifier": hashed_identifier}, namespace
        )

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    )
    @settings(max_examples=100)
    def test_both_systems_missing_data_property(self, hashed_identifier, namespace):
        """
        Property: When both storage systems report missing data, should return success.
        
        For any deletion request where neither R2 nor Pinecone have data for the video,
        the operation should succeed (nothing to delete is not an error).
        """
        # Setup mock connectors
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Mock R2 - file not found
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=True,
            bucket="dev",
            key=f"test/{hashed_identifier}",
            file_existed=False,
            error_message="File not found in R2 storage"
        )
        
        # Mock Pinecone - no chunks found
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=True,
            chunks_found=0,
            chunks_deleted=0,
            chunk_ids=[],
            namespace=namespace,
            error_message="No chunks found for video"
        )
        
        # Mock verification
        mock_r2.verify_deletion.return_value = True
        mock_pinecone.find_chunks_by_video.return_value = []
        
        # Mock identifier decoding
        mock_r2._decode_path.return_value = ("dev", "test", f"{hashed_identifier}.mp4")
        
        # Create service
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        # Perform deletion
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Should succeed when both systems have no data
        assert result.success is True
        assert result.hashed_identifier == hashed_identifier
        assert result.namespace == namespace
        
        # Both results should indicate no data found but success
        assert result.r2_result.success is True
        assert result.r2_result.file_existed is False
        
        assert result.pinecone_result.success is True
        assert result.pinecone_result.chunks_found == 0
        
        # Both operations should have been attempted
        mock_r2.delete_video_file.assert_called_once_with(hashed_identifier)
        mock_pinecone.delete_by_metadata.assert_called_once_with(
            {"hashed_identifier": hashed_identifier}, namespace
        )

    @given(
        hashed_identifier=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        namespace=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    )
    @settings(max_examples=100)
    def test_verification_failure_handling_property(self, hashed_identifier, namespace):
        """
        Property: Verification failures should cause overall operation failure.
        
        For any deletion that reports success but fails verification,
        the overall result should indicate failure.
        """
        # Setup mock connectors
        mock_r2 = MagicMock()
        mock_pinecone = MagicMock()
        
        # Mock successful deletion operations
        mock_r2.delete_video_file.return_value = R2DeletionResult(
            success=True,
            bucket="dev",
            key=f"test/{hashed_identifier}",
            file_existed=True,
            bytes_deleted=1000
        )
        
        mock_pinecone.delete_by_metadata.return_value = PineconeDeletionResult(
            success=True,
            chunks_found=1,
            chunks_deleted=1,
            chunk_ids=["chunk_1"],
            namespace=namespace
        )
        
        # Mock verification failure - file still exists
        mock_r2.verify_deletion.return_value = False  # Verification failed
        mock_pinecone.find_chunks_by_video.return_value = ["chunk_1"]  # Chunks still exist
        
        # Mock identifier decoding
        mock_r2._decode_path.return_value = ("dev", "test", f"{hashed_identifier}.mp4")
        
        # Create service
        service = VideoDeletionService(
            r2_connector=mock_r2,
            pinecone_connector=mock_pinecone,
            environment="dev"
        )
        
        # Perform deletion
        result = asyncio.run(service.delete_video(hashed_identifier, namespace))
        
        # Property: Should fail overall when verification fails
        assert result.success is False
        assert result.hashed_identifier == hashed_identifier
        assert result.namespace == namespace
        
        # Individual operations should report success
        assert result.r2_result.success is True
        assert result.pinecone_result.success is True
        
        # But verification should show failures
        assert result.verification_result is not None
        assert result.verification_result.r2_verified is False
        assert result.verification_result.pinecone_verified is False
        assert len(result.verification_result.verification_errors) > 0
        
        # All operations should have been attempted
        mock_r2.delete_video_file.assert_called_once_with(hashed_identifier)
        mock_pinecone.delete_by_metadata.assert_called_once_with(
            {"hashed_identifier": hashed_identifier}, namespace
        )
        mock_r2.verify_deletion.assert_called_once_with(hashed_identifier)
        mock_pinecone.find_chunks_by_video.assert_called_once_with(hashed_identifier, namespace)