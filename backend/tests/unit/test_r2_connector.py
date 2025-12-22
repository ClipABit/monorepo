import pytest
import base64
from botocore.exceptions import ClientError
from database.r2_connector import R2Connector


class TestR2ConnectorInitialization:
    """Test connector initialization."""

    def test_initializes_with_credentials(self, mock_r2_connector):
        """Verify connector initializes with required parameters."""
        connector, mock_client, mock_boto3 = mock_r2_connector

        assert connector.bucket_name == "dev"
        assert connector.endpoint_url == "https://test-account.r2.cloudflarestorage.com"
        mock_boto3.client.assert_called_once()
        assert connector.s3_client == mock_client


class TestUploadVideo:
    """Test video upload operations."""

    def test_upload_video_success(self, mock_r2_connector, sample_video_5s):
        """Verify successful video upload."""
        connector, mock_client, _ = mock_r2_connector
        video_data = sample_video_5s.read_bytes()
        filename = "test video.mp4"
        namespace = "test-namespace"

        success, identifier = connector.upload_video(video_data, filename, namespace)

        assert success is True
        assert identifier is not None
        
        # Verify put_object called correctly
        args = mock_client.put_object.call_args[1]
        assert args['Bucket'] == "dev"
        assert args['Body'] == video_data
        assert args['ContentType'] == "video/mp4"
        assert args['Key'].startswith("test-namespace/")
        assert args['Key'].endswith("_test_video.mp4")

    def test_upload_video_client_error(self, mock_r2_connector):
        """Verify upload handles client errors."""
        connector, mock_client, _ = mock_r2_connector
        mock_client.put_object.side_effect = ClientError({'Error': {'Code': '500'}}, 'put_object')

        success, identifier = connector.upload_video(b"data", "test.mp4")

        assert success is False
        assert "500" in identifier  # Error message returned in identifier field


class TestFetchVideo:
    """Test video fetch operations."""

    def test_fetch_video_success(self, mock_r2_connector, mocker):
        """Verify successful video fetch."""
        connector, mock_client, _ = mock_r2_connector
        
        # Setup mock response
        mock_body = mocker.MagicMock()
        mock_body.read.return_value = b"video-content"
        mock_client.get_object.return_value = {'Body': mock_body}

        # Create valid identifier
        identifier = base64.urlsafe_b64encode(b"dev/ns/vid.mp4").decode('utf-8')

        result = connector.fetch_video(identifier)

        assert result == b"video-content"
        mock_client.get_object.assert_called_once_with(
            Bucket="dev",
            Key="ns/vid.mp4"
        )

    def test_fetch_video_invalid_identifier(self, mock_r2_connector):
        """Verify fetch fails with invalid identifier."""
        connector, _, _ = mock_r2_connector
        result = connector.fetch_video("bad-id")
        assert result is None

    def test_fetch_video_client_error(self, mock_r2_connector):
        """Verify fetch handles client errors."""
        connector, mock_client, _ = mock_r2_connector
        mock_client.get_object.side_effect = ClientError({'Error': {'Code': '404'}}, 'get_object')
        
        identifier = base64.urlsafe_b64encode(b"dev/ns/vid.mp4").decode('utf-8')
        result = connector.fetch_video(identifier)
        
        assert result is None


class TestGeneratePresignedURL:
    """Test presigned URL generation."""

    def test_generate_url_success(self, mock_r2_connector):
        """Verify successful URL generation."""
        connector, mock_client, _ = mock_r2_connector
        mock_client.generate_presigned_url.return_value = "https://signed-url"
        
        identifier = base64.urlsafe_b64encode(b"dev/ns/vid.mp4").decode('utf-8')
        url = connector.generate_presigned_url(identifier)

        assert url == "https://signed-url"
        mock_client.generate_presigned_url.assert_called_once()
        args = mock_client.generate_presigned_url.call_args
        assert args[0][0] == 'get_object'
        assert args[1]['Params']['Bucket'] == "dev"
        assert args[1]['Params']['Key'] == "ns/vid.mp4"

    def test_generate_url_bucket_mismatch(self, mock_r2_connector):
        """Verify URL generation fails on bucket mismatch."""
        connector, _, _ = mock_r2_connector
        identifier = base64.urlsafe_b64encode(b"staging/ns/vid.mp4").decode('utf-8')
        
        url = connector.generate_presigned_url(identifier)
        assert url is None


class TestFetchAllVideoData:
    """Test listing videos."""

    def test_fetch_all_success(self, mock_r2_connector, mocker):
        """Verify listing videos works."""
        connector, mock_client, _ = mock_r2_connector
        
        # Mock paginator
        mock_paginator = mocker.MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        
        # Mock page content
        mock_paginator.paginate.return_value = [{
            'Contents': [
                {'Key': 'ns/vid1.mp4'},
                {'Key': 'ns/vid2.mp4'}
            ]
        }]
        
        mock_client.generate_presigned_url.return_value = "http://url"

        results = connector.fetch_all_video_data("ns")

        assert len(results) == 2
        assert results[0]['file_name'] == "vid1.mp4"
        assert results[1]['file_name'] == "vid2.mp4"
        assert results[0]['presigned_url'] == "http://url"


class TestDeleteVideo:
    """Test video deletion operations."""

    def test_delete_video_success(self, mock_r2_connector):
        """Verify successful video deletion."""
        connector, mock_client, _ = mock_r2_connector
        
        identifier = base64.urlsafe_b64encode(b"dev/test-namespace/video.mp4").decode('utf-8')
        
        result = connector.delete_video(identifier)

        assert result is True
        mock_client.delete_object.assert_called_once_with(
            Bucket="dev",
            Key="test-namespace/video.mp4"
        )

    def test_delete_video_invalid_identifier(self, mock_r2_connector):
        """Verify deletion fails with invalid identifier."""
        connector, mock_client, _ = mock_r2_connector
        
        result = connector.delete_video("invalid-identifier")

        assert result is False
        mock_client.delete_object.assert_not_called()

    def test_delete_video_bucket_mismatch(self, mock_r2_connector):
        """Verify deletion fails if bucket name doesn't match."""
        connector, mock_client, _ = mock_r2_connector
        
        # Identifier for 'staging' bucket but connector is 'test'
        identifier = base64.urlsafe_b64encode(b"staging/test-namespace/video.mp4").decode('utf-8')
        
        result = connector.delete_video(identifier)

        assert result is False
        mock_client.delete_object.assert_not_called()

    def test_delete_video_client_error(self, mock_r2_connector):
        """Verify deletion handles client errors gracefully."""
        connector, mock_client, _ = mock_r2_connector
        
        identifier = base64.urlsafe_b64encode(b"dev/test-namespace/video.mp4").decode('utf-8')
        
        # Simulate ClientError
        error_response = {'Error': {'Code': '500', 'Message': 'Internal Error'}}
        mock_client.delete_object.side_effect = ClientError(error_response, 'delete_object')
        
        result = connector.delete_video(identifier)

        assert result is False
        mock_client.delete_object.assert_called_once()
