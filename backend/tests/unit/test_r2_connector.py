"""
Tests for R2Connector class.

This module tests both existing functionality and new deletion capabilities
of the R2Connector class, including property-based tests for deletion operations.
"""

import pytest
from botocore.exceptions import ClientError
from hypothesis import given, strategies as st
from database.r2_connector import R2Connector


class TestR2ConnectorInitialization:
    """Test R2Connector initialization."""

    def test_initializes_with_required_parameters(self):
        """Verify connector initializes with required parameters."""
        connector = R2Connector(
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
            environment="dev"
        )
        
        assert connector.bucket_name == "dev"
        assert "test-account" in connector.endpoint_url


class TestR2ConnectorDeletion:
    """Test R2Connector deletion functionality."""

    @pytest.fixture
    def mock_r2_connector(self, mocker):
        """Create a mock R2Connector with mocked boto3 client."""
        # Mock boto3.client to return a mock S3 client
        mock_s3_client = mocker.MagicMock()
        mocker.patch('boto3.client', return_value=mock_s3_client)
        
        connector = R2Connector(
            account_id="test-account",
            access_key_id="test-key", 
            secret_access_key="test-secret",
            environment="test-bucket"
        )
        
        return connector, mock_s3_client

    def test_delete_nonexistent_file_returns_success(self, mock_r2_connector):
        """Test that deleting a non-existent file returns success with appropriate flags."""
        connector, mock_s3_client = mock_r2_connector
        
        # Mock head_object to raise 404 error (file doesn't exist)
        mock_s3_client.head_object.side_effect = ClientError(
            error_response={'Error': {'Code': '404'}},
            operation_name='HeadObject'
        )
        
        # Create a valid identifier for a non-existent file
        identifier = connector._encode_path("test-bucket", "test-namespace", "nonexistent.mp4")
        
        result = connector.delete_video_file(identifier)
        
        assert result.success is True
        assert result.file_existed is False
        assert result.bucket == "test-bucket"
        assert "nonexistent.mp4" in result.key
        assert "File not found" in result.error_message
        
        # Verify head_object was called but delete_object was not
        mock_s3_client.head_object.assert_called_once()
        mock_s3_client.delete_object.assert_not_called()

    def test_delete_existing_file_success(self, mock_r2_connector):
        """Test successful deletion of an existing file."""
        connector, mock_s3_client = mock_r2_connector
        
        # Mock head_object to return file info (file exists)
        mock_s3_client.head_object.return_value = {'ContentLength': 1024}
        
        # Mock delete_object to succeed
        mock_s3_client.delete_object.return_value = {}
        
        # Create identifier for the file
        identifier = connector._encode_path("test-bucket", "test-namespace", "test-video.mp4")
        
        result = connector.delete_video_file(identifier)
        
        assert result.success is True
        assert result.file_existed is True
        assert result.bucket == "test-bucket"
        assert result.key == "test-namespace/test-video.mp4"
        assert result.bytes_deleted == 1024
        assert result.error_message is None
        
        # Verify both head_object and delete_object were called
        mock_s3_client.head_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test-namespace/test-video.mp4"
        )
        mock_s3_client.delete_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test-namespace/test-video.mp4"
        )

    def test_delete_with_invalid_identifier(self, mock_r2_connector):
        """Test deletion with invalid identifier format."""
        connector, mock_s3_client = mock_r2_connector
        
        invalid_identifier = "invalid-identifier-format"
        
        result = connector.delete_video_file(invalid_identifier)
        
        assert result.success is False
        assert result.file_existed is False
        assert "Invalid hashed identifier" in result.error_message
        
        # Verify no S3 calls were made
        mock_s3_client.head_object.assert_not_called()
        mock_s3_client.delete_object.assert_not_called()

    def test_delete_with_s3_error(self, mock_r2_connector):
        """Test deletion when S3 operation fails."""
        connector, mock_s3_client = mock_r2_connector
        
        # Mock head_object to raise a non-404 error
        mock_s3_client.head_object.side_effect = ClientError(
            error_response={'Error': {'Code': '500', 'Message': 'Internal Server Error'}},
            operation_name='HeadObject'
        )
        
        identifier = connector._encode_path("test-bucket", "test-namespace", "test-video.mp4")
        
        result = connector.delete_video_file(identifier)
        
        assert result.success is False
        assert result.file_existed is False
        assert "Error deleting video from R2" in result.error_message

    def test_verify_deletion_success(self, mock_r2_connector):
        """Test successful deletion verification."""
        connector, mock_s3_client = mock_r2_connector
        
        # Mock head_object to raise 404 (file doesn't exist - deletion verified)
        mock_s3_client.head_object.side_effect = ClientError(
            error_response={'Error': {'Code': '404'}},
            operation_name='HeadObject'
        )
        
        identifier = connector._encode_path("test-bucket", "test-namespace", "deleted-video.mp4")
        
        result = connector.verify_deletion(identifier)
        
        assert result is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test-namespace/deleted-video.mp4"
        )

    def test_verify_deletion_file_still_exists(self, mock_r2_connector):
        """Test deletion verification when file still exists."""
        connector, mock_s3_client = mock_r2_connector
        
        # Mock head_object to succeed (file still exists)
        mock_s3_client.head_object.return_value = {'ContentLength': 1024}
        
        identifier = connector._encode_path("test-bucket", "test-namespace", "existing-video.mp4")
        
        result = connector.verify_deletion(identifier)
        
        assert result is False

    def test_verify_deletion_invalid_identifier(self, mock_r2_connector):
        """Test deletion verification with invalid identifier."""
        connector, mock_s3_client = mock_r2_connector
        
        invalid_identifier = "invalid-identifier"
        
        result = connector.verify_deletion(invalid_identifier)
        
        assert result is False
        
        # Verify no S3 calls were made
        mock_s3_client.head_object.assert_not_called()


class TestR2DeletionProperties:
    """Property-based tests for R2 deletion functionality."""

    @given(
        namespace=st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z'))),
        filename=st.text(min_size=5, max_size=20, alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z'))).map(lambda x: x + '.mp4'),
        content_size=st.integers(min_value=1, max_value=10000)
    )
    def test_property_complete_r2_deletion(self, namespace, filename, content_size):
        """
        Property 1: Complete R2 Deletion
        For any valid hashed_identifier with corresponding R2 data, 
        deleting the video should result in the video file no longer existing in R2 storage.
        
        Feature: video-deletion, Property 1: Complete R2 Deletion
        Validates: Requirements 1.1
        """
        from unittest.mock import MagicMock, patch
        
        # Create mock S3 client for this test iteration
        mock_s3_client = MagicMock()
        
        with patch('boto3.client', return_value=mock_s3_client):
            connector = R2Connector(
                account_id="test-account",
                access_key_id="test-key",
                secret_access_key="test-secret", 
                environment="test-bucket"
            )
            
            # Mock head_object to return file info (file exists)
            mock_s3_client.head_object.return_value = {'ContentLength': content_size}
            
            # Mock delete_object to succeed
            mock_s3_client.delete_object.return_value = {}
            
            # Create identifier
            identifier = connector._encode_path("test-bucket", namespace, filename)
            
            # Delete the file
            result = connector.delete_video_file(identifier)
            
            # Verify deletion was successful
            assert result.success is True
            assert result.file_existed is True
            assert result.bytes_deleted == content_size
            
            # Verify the correct S3 calls were made
            expected_key = f"{namespace}/{filename}"
            mock_s3_client.head_object.assert_called_with(
                Bucket="test-bucket",
                Key=expected_key
            )
            mock_s3_client.delete_object.assert_called_with(
                Bucket="test-bucket",
                Key=expected_key
            )
            
            # Now test verification - mock head_object to raise 404 (file deleted)
            mock_s3_client.head_object.side_effect = ClientError(
                error_response={'Error': {'Code': '404'}},
                operation_name='HeadObject'
            )
            
            # Verify file no longer exists
            verification_result = connector.verify_deletion(identifier)
            assert verification_result is True, "File should no longer exist after deletion"
