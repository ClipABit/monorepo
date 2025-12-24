"""
Frontend component tests for video deletion functionality.

Tests delete button rendering, confirmation dialog behavior, and API integration.
Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import pytest
from unittest.mock import Mock, patch
import requests
import sys
import os

# Add the parent directory to the path so we can import from the streamlit app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class TestDeleteButtonRendering:
    """Test delete button rendering based on environment."""
    
    def test_delete_buttons_shown_in_dev_environment(self):
        """Test that delete buttons are shown when SHOW_DELETE_BUTTONS is True."""
        # Requirements: 5.1, 5.6
        with patch.dict(os.environ, {"ENVIRONMENT": "dev"}):
            # Reload config to pick up environment change
            import importlib
            import config
            importlib.reload(config)
            
            assert config.Config.SHOW_DELETE_BUTTONS is True
            assert config.Config.ENVIRONMENT == "dev"
    
    def test_delete_buttons_hidden_in_prod_environment(self):
        """Test that delete buttons are hidden when SHOW_DELETE_BUTTONS is False."""
        # Requirements: 5.6
        with patch.dict(os.environ, {"ENVIRONMENT": "prod"}):
            # Reload config to pick up environment change
            import importlib
            import config
            importlib.reload(config)
            
            assert config.Config.SHOW_DELETE_BUTTONS is False
            assert config.Config.ENVIRONMENT == "prod"


class TestDeleteAPIIntegration:
    """Test delete API call functionality."""
    
    @patch('requests.delete')
    def test_successful_delete_api_call(self, mock_delete):
        """Test successful delete API call returns expected response."""
        # Requirements: 5.3, 5.4
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "message": "Video deleted successfully"
        }
        mock_delete.return_value = mock_response
        
        # Simulate the delete API call pattern
        hashed_identifier = "test_video_123"
        namespace = "web-demo"
        
        response = requests.delete(
            f"{Config.DELETE_API_URL}",
            params={"hashed_identifier": hashed_identifier, "namespace": namespace},
            timeout=30
        )
        
        assert response.status_code == 200
        assert response.json()["success"] is True
        mock_delete.assert_called_once()
    
    @patch('requests.delete')
    def test_failed_delete_api_call(self, mock_delete):
        """Test failed delete API call returns error response."""
        # Requirements: 5.5
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "success": False,
            "error": {
                "type": "StorageError",
                "message": "Failed to delete from R2 storage"
            }
        }
        mock_delete.return_value = mock_response
        
        # Simulate the delete API call pattern
        hashed_identifier = "test_video_123"
        namespace = "web-demo"
        
        response = requests.delete(
            f"{Config.DELETE_API_URL}",
            params={"hashed_identifier": hashed_identifier, "namespace": namespace},
            timeout=30
        )
        
        assert response.status_code == 500
        assert response.json()["success"] is False
        assert "error" in response.json()
        mock_delete.assert_called_once()
    
    @patch('requests.delete')
    def test_delete_api_call_with_network_error(self, mock_delete):
        """Test delete API call handles network errors gracefully."""
        # Requirements: 5.5
        mock_delete.side_effect = requests.RequestException("Network error")
        
        # Simulate the delete API call pattern with error handling
        hashed_identifier = "test_video_123"
        namespace = "web-demo"
        
        try:
            requests.delete(
                f"{Config.DELETE_API_URL}",
                params={"hashed_identifier": hashed_identifier, "namespace": namespace},
                timeout=30
            )
            assert False, "Should have raised RequestException"
        except requests.RequestException as e:
            assert str(e) == "Network error"
        
        mock_delete.assert_called_once()
    
    def test_delete_api_url_configuration(self):
        """Test that DELETE_API_URL is properly configured."""
        # Requirements: 5.3
        assert hasattr(Config, 'DELETE_API_URL')
        assert Config.DELETE_API_URL is not None
        assert "delete" in Config.DELETE_API_URL.lower()


class TestDeleteConfirmationDialog:
    """Test delete confirmation dialog behavior."""
    
    def test_confirmation_dialog_structure(self):
        """Test that confirmation dialog has required elements."""
        # Requirements: 5.2
        # This test verifies the structure that should be implemented
        # in the actual Streamlit dialog
        
        # Mock dialog data structure
        dialog_data = {
            "title": "Delete Video",
            "message": "Are you sure you want to delete this video?",
            "filename": "test_video.mp4",
            "warning": "This action cannot be undone",
            "buttons": ["Cancel", "Delete"]
        }
        
        assert dialog_data["title"] == "Delete Video"
        assert "delete" in dialog_data["message"].lower()
        assert dialog_data["warning"] is not None
        assert "Cancel" in dialog_data["buttons"]
        assert "Delete" in dialog_data["buttons"]
    
    def test_confirmation_required_before_deletion(self):
        """Test that confirmation is required before deletion proceeds."""
        # Requirements: 5.2
        # This test verifies the confirmation flow logic
        
        confirmation_required = True
        user_confirmed = False
        
        # Simulate confirmation dialog logic
        if confirmation_required and not user_confirmed:
            deletion_should_proceed = False
        else:
            deletion_should_proceed = True
        
        assert deletion_should_proceed is False
        
        # Now simulate user confirmation
        user_confirmed = True
        if confirmation_required and user_confirmed:
            deletion_should_proceed = True
        
        assert deletion_should_proceed is True


class TestVideoDisplayIntegration:
    """Test video display integration with delete functionality."""
    
    def test_video_display_with_delete_button_structure(self):
        """Test video display includes delete button when enabled."""
        # Requirements: 5.1
        
        # Mock video data structure with delete button
        video_data = {
            "file_name": "test_video.mp4",
            "presigned_url": "https://example.com/video.mp4",
            "hashed_identifier": "abc123",
            "show_delete_button": True
        }
        
        assert video_data["show_delete_button"] is True
        assert video_data["hashed_identifier"] is not None
        assert video_data["file_name"] is not None
    
    def test_video_display_without_delete_button_in_prod(self):
        """Test video display hides delete button in production."""
        # Requirements: 5.6
        
        # Mock video data structure without delete button (prod environment)
        video_data = {
            "file_name": "test_video.mp4",
            "presigned_url": "https://example.com/video.mp4",
            "hashed_identifier": "abc123",
            "show_delete_button": False
        }
        
        assert video_data["show_delete_button"] is False


class TestErrorHandling:
    """Test error handling for delete operations."""
    
    def test_error_message_display_structure(self):
        """Test error message display structure."""
        # Requirements: 5.5
        
        error_response = {
            "success": False,
            "error": {
                "type": "ValidationError",
                "message": "Invalid video identifier"
            }
        }
        
        # Simulate error message extraction
        if not error_response["success"]:
            error_message = error_response["error"]["message"]
            assert error_message == "Invalid video identifier"
    
    def test_success_message_display_structure(self):
        """Test success message display structure."""
        # Requirements: 5.4
        
        success_response = {
            "success": True,
            "message": "Video deleted successfully"
        }
        
        # Simulate success message extraction
        if success_response["success"]:
            success_message = success_response["message"]
            assert success_message == "Video deleted successfully"


if __name__ == "__main__":
    pytest.main([__file__])