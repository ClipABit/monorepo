from database.job_store_connector import JobStoreConnector


class TestJobCreation:
    """Test job creation operations."""

    def test_create_job_stores_data(self, mock_modal_dict):
        """Verify job creation stores data in dict."""
        connector = JobStoreConnector("test-jobs")

        result = connector.create_job("job-123", {
            "job_id": "job-123",
            "status": "processing",
            "filename": "test.mp4"
        })

        assert result is True
        assert "job-123" in mock_modal_dict
        assert mock_modal_dict["job-123"]["status"] == "processing"

    def test_create_multiple_jobs(self, mock_modal_dict):
        """Verify multiple jobs can be created."""
        connector = JobStoreConnector("test-jobs")

        connector.create_job("job-1", {"status": "processing"})
        connector.create_job("job-2", {"status": "completed"})
        connector.create_job("job-3", {"status": "failed"})

        assert len(mock_modal_dict) == 3
        assert "job-1" in mock_modal_dict
        assert "job-2" in mock_modal_dict
        assert "job-3" in mock_modal_dict


class TestJobRetrieval:
    """Test job retrieval operations."""

    def test_get_existing_job(self, mock_modal_dict):
        """Verify retrieval of existing job."""
        connector = JobStoreConnector("test-jobs")
        connector.create_job("job-123", {"status": "processing", "filename": "test.mp4"})

        job_data = connector.get_job("job-123")

        assert job_data is not None
        assert job_data["status"] == "processing"
        assert job_data["filename"] == "test.mp4"

    def test_get_nonexistent_job_returns_none(self, mock_modal_dict):
        """Verify retrieval of non-existent job returns None."""
        connector = JobStoreConnector("test-jobs")

        job_data = connector.get_job("nonexistent")

        assert job_data is None

    def test_job_exists_check(self, mock_modal_dict):
        """Verify job existence check."""
        connector = JobStoreConnector("test-jobs")
        connector.create_job("job-123", {"status": "processing"})

        assert connector.job_exists("job-123") is True
        assert connector.job_exists("nonexistent") is False


class TestJobUpdate:
    """Test job update operations."""

    def test_update_job_merges_data(self, mock_modal_dict):
        """Verify update merges with existing data."""
        connector = JobStoreConnector("test-jobs")
        connector.create_job("job-123", {
            "status": "processing",
            "filename": "test.mp4",
            "size": 1024
        })

        result = connector.update_job("job-123", {
            "status": "completed",
            "chunks": 5
        })

        assert result is True
        job_data = connector.get_job("job-123")
        assert job_data["status"] == "completed"  # Updated
        assert job_data["filename"] == "test.mp4"  # Preserved
        assert job_data["size"] == 1024  # Preserved
        assert job_data["chunks"] == 5  # Added

    def test_update_nonexistent_job_returns_false(self, mock_modal_dict):
        """Verify updating non-existent job returns False."""
        connector = JobStoreConnector("test-jobs")

        result = connector.update_job("nonexistent", {"status": "completed"})

        assert result is False

    def test_set_job_completed(self, mock_modal_dict):
        """Verify helper method for marking job completed."""
        connector = JobStoreConnector("test-jobs")
        connector.create_job("job-123", {"status": "processing"})

        result = connector.set_job_completed("job-123", {
            "chunks": 10,
            "total_frames": 100
        })

        assert result is True
        job_data = connector.get_job("job-123")
        assert job_data["status"] == "completed"
        assert job_data["chunks"] == 10

    def test_set_job_failed(self, mock_modal_dict):
        """Verify helper method for marking job failed."""
        connector = JobStoreConnector("test-jobs")
        connector.create_job("job-123", {"status": "processing"})

        result = connector.set_job_failed("job-123", "Processing error")

        assert result is True
        job_data = connector.get_job("job-123")
        assert job_data["status"] == "failed"
        assert job_data["error"] == "Processing error"


class TestJobDeletion:
    """Test job deletion operations."""

    def test_delete_existing_job(self, mock_modal_dict):
        """Verify job deletion."""
        connector = JobStoreConnector("test-jobs")
        connector.create_job("job-123", {"status": "processing"})

        result = connector.delete_job("job-123")

        assert result is True
        assert "job-123" not in mock_modal_dict

    def test_delete_nonexistent_job_returns_false(self, mock_modal_dict):
        """Verify deleting non-existent job returns False."""
        connector = JobStoreConnector("test-jobs")

        result = connector.delete_job("nonexistent")

        assert result is False


class TestConcurrentOperations:
    """Test multiple operations on same connector."""

    def test_full_job_lifecycle(self, mock_modal_dict):
        """Test complete job lifecycle: create -> update -> complete -> delete."""
        connector = JobStoreConnector("test-jobs")

        # Create
        connector.create_job("job-123", {"status": "processing", "filename": "test.mp4"})
        assert connector.job_exists("job-123")

        # Update
        connector.update_job("job-123", {"progress": 50})
        assert connector.get_job("job-123")["progress"] == 50

        # Complete
        connector.set_job_completed("job-123", {"chunks": 5})
        assert connector.get_job("job-123")["status"] == "completed"

        # Delete
        connector.delete_job("job-123")
        assert not connector.job_exists("job-123")

    def test_multiple_connectors_share_state(self, mock_modal_dict):
        """Verify multiple connectors access same underlying dict."""
        connector1 = JobStoreConnector("test-jobs")
        connector2 = JobStoreConnector("test-jobs")

        connector1.create_job("job-123", {"status": "processing"})

        # connector2 should see the job
        job_data = connector2.get_job("job-123")
        assert job_data is not None
        assert job_data["status"] == "processing"
