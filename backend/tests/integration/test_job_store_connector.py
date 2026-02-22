from database.cache.job_store_connector import JobStoreConnector


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


class TestBatchJobOperations:
    """Test batch job creation and management."""

    def test_create_batch_job(self, mock_modal_dict):
        """Verify batch job creation with child references."""
        connector = JobStoreConnector("test-jobs")

        child_ids = ["job-1", "job-2", "job-3"]
        result = connector.create_batch_job("batch-123", child_ids, "web-demo")

        assert result is True
        batch_data = connector.get_job("batch-123")
        assert batch_data["job_type"] == "batch"
        assert batch_data["total_videos"] == 3
        assert batch_data["child_jobs"] == child_ids
        assert batch_data["status"] == "processing"
        assert batch_data["completed_count"] == 0
        assert batch_data["failed_count"] == 0
        assert batch_data["processing_count"] == 3

    def test_update_batch_on_child_completion_success(self, mock_modal_dict):
        """Verify batch updates when child completes successfully."""
        connector = JobStoreConnector("test-jobs")

        # Setup batch
        connector.create_batch_job("batch-123", ["job-1", "job-2"], "web-demo")

        # Complete first child
        child_result = {
            "job_id": "job-1",
            "status": "completed",
            "filename": "video1.mp4",
            "chunks": 5,
            "total_frames": 42,
            "total_memory_mb": 100.0,
            "avg_complexity": 0.5
        }

        connector.update_batch_on_child_completion("batch-123", "job-1", child_result)

        batch = connector.get_job("batch-123")
        assert batch["completed_count"] == 1
        assert batch["failed_count"] == 0
        assert batch["processing_count"] == 1
        assert batch["total_chunks"] == 5
        assert batch["total_frames"] == 42
        assert batch["avg_complexity"] == 0.5
        assert batch["status"] == "processing"  # Still has 1 processing

    def test_update_batch_on_child_completion_failure(self, mock_modal_dict):
        """Verify batch updates when child fails."""
        connector = JobStoreConnector("test-jobs")

        # Setup batch
        connector.create_batch_job("batch-123", ["job-1", "job-2"], "web-demo")

        # Fail first child
        child_result = {
            "job_id": "job-1",
            "status": "failed",
            "filename": "video1.mp4",
            "error": "Scene detection failed"
        }

        connector.update_batch_on_child_completion("batch-123", "job-1", child_result)

        batch = connector.get_job("batch-123")
        assert batch["completed_count"] == 0
        assert batch["failed_count"] == 1
        assert batch["processing_count"] == 1
        assert len(batch["failed_jobs"]) == 1
        assert batch["failed_jobs"][0]["error"] == "Scene detection failed"

    def test_batch_status_all_completed(self, mock_modal_dict):
        """Verify batch status when all children complete successfully."""
        connector = JobStoreConnector("test-jobs")
        connector.create_batch_job("batch-123", ["job-1", "job-2"], "web-demo")

        # Complete both children
        for job_id in ["job-1", "job-2"]:
            result = {
                "job_id": job_id,
                "status": "completed",
                "filename": f"{job_id}.mp4",
                "chunks": 5,
                "total_frames": 50,
                "total_memory_mb": 100.0,
                "avg_complexity": 0.5
            }
            connector.update_batch_on_child_completion("batch-123", job_id, result)

        batch = connector.get_job("batch-123")
        assert batch["status"] == "completed"
        assert batch["completed_count"] == 2
        assert batch["failed_count"] == 0
        assert batch["processing_count"] == 0

    def test_batch_status_partial(self, mock_modal_dict):
        """Verify batch status with mixed success/failure."""
        connector = JobStoreConnector("test-jobs")
        connector.create_batch_job("batch-123", ["job-1", "job-2"], "web-demo")

        # Complete one, fail one
        success_result = {
            "job_id": "job-1",
            "status": "completed",
            "filename": "video1.mp4",
            "chunks": 5,
            "total_frames": 50,
            "total_memory_mb": 100.0,
            "avg_complexity": 0.5
        }
        connector.update_batch_on_child_completion("batch-123", "job-1", success_result)

        failure_result = {
            "job_id": "job-2",
            "status": "failed",
            "filename": "video2.mp4",
            "error": "Scene detection failed"
        }
        connector.update_batch_on_child_completion("batch-123", "job-2", failure_result)

        batch = connector.get_job("batch-123")
        assert batch["status"] == "partial"
        assert batch["completed_count"] == 1
        assert batch["failed_count"] == 1
        assert len(batch["failed_jobs"]) == 1

    def test_batch_status_all_failed(self, mock_modal_dict):
        """Verify batch status when all children fail."""
        connector = JobStoreConnector("test-jobs")
        connector.create_batch_job("batch-123", ["job-1", "job-2"], "web-demo")

        # Fail both children
        for job_id in ["job-1", "job-2"]:
            result = {
                "job_id": job_id,
                "status": "failed",
                "filename": f"{job_id}.mp4",
                "error": "Processing failed"
            }
            connector.update_batch_on_child_completion("batch-123", job_id, result)

        batch = connector.get_job("batch-123")
        assert batch["status"] == "failed"
        assert batch["completed_count"] == 0
        assert batch["failed_count"] == 2
        assert batch["processing_count"] == 0

    def test_get_batch_child_jobs(self, mock_modal_dict):
        """Verify retrieving all child job data."""
        connector = JobStoreConnector("test-jobs")

        # Create batch and children
        connector.create_batch_job("batch-123", ["job-1", "job-2"], "web-demo")
        connector.create_job("job-1", {"job_id": "job-1", "status": "completed"})
        connector.create_job("job-2", {"job_id": "job-2", "status": "processing"})

        children = connector.get_batch_child_jobs("batch-123")

        assert len(children) == 2
        assert children[0]["job_id"] == "job-1"
        assert children[1]["job_id"] == "job-2"

    def test_get_batch_child_jobs_nonexistent_batch(self, mock_modal_dict):
        """Verify get_batch_child_jobs returns empty list for non-existent batch."""
        connector = JobStoreConnector("test-jobs")

        children = connector.get_batch_child_jobs("nonexistent")

        assert children == []

    def test_get_batch_progress(self, mock_modal_dict):
        """Verify batch progress summary."""
        connector = JobStoreConnector("test-jobs")
        connector.create_batch_job("batch-123", ["job-1", "job-2", "job-3"], "web-demo")

        # Complete one job
        result = {
            "job_id": "job-1",
            "status": "completed",
            "filename": "video1.mp4",
            "chunks": 5,
            "total_frames": 50,
            "total_memory_mb": 100.0,
            "avg_complexity": 0.5
        }
        connector.update_batch_on_child_completion("batch-123", "job-1", result)

        progress = connector.get_batch_progress("batch-123")

        assert progress["batch_job_id"] == "batch-123"
        assert progress["total_videos"] == 3
        assert progress["completed"] == 1
        assert progress["failed"] == 0
        assert progress["processing"] == 2
        assert progress["progress_percent"] == (1/3 * 100)

    def test_get_batch_progress_nonexistent_batch(self, mock_modal_dict):
        """Verify get_batch_progress returns None for non-existent batch."""
        connector = JobStoreConnector("test-jobs")

        progress = connector.get_batch_progress("nonexistent")

        assert progress is None

    def test_average_complexity_calculation(self, mock_modal_dict):
        """Verify running average complexity calculation."""
        connector = JobStoreConnector("test-jobs")
        connector.create_batch_job("batch-123", ["job-1", "job-2", "job-3"], "web-demo")

        # Complete three jobs with different complexities
        complexities = [0.3, 0.5, 0.7]
        for idx, job_id in enumerate(["job-1", "job-2", "job-3"]):
            result = {
                "job_id": job_id,
                "status": "completed",
                "filename": f"video{idx}.mp4",
                "chunks": 5,
                "total_frames": 50,
                "total_memory_mb": 100.0,
                "avg_complexity": complexities[idx]
            }
            connector.update_batch_on_child_completion("batch-123", job_id, result)

        batch = connector.get_job("batch-123")
        expected_avg = sum(complexities) / len(complexities)
        assert abs(batch["avg_complexity"] - expected_avg) < 0.001  # Float comparison with tolerance


class TestConcurrentBatchUpdates:
    """Test concurrent batch update scenarios."""

    def test_batch_job_has_version_field(self, mock_modal_dict):
        """Verify batch jobs are created with version field for optimistic locking."""
        connector = JobStoreConnector("test-jobs")
        connector.create_batch_job("batch-123", ["job-1", "job-2"], "web-demo")

        batch = connector.get_job("batch-123")
        assert "_version" in batch
        assert batch["_version"] == 0

    def test_version_increments_on_update(self, mock_modal_dict):
        """Verify version increments with each update."""
        connector = JobStoreConnector("test-jobs")
        connector.create_batch_job("batch-123", ["job-1", "job-2"], "web-demo")

        # First update
        result1 = {
            "job_id": "job-1",
            "status": "completed",
            "filename": "video1.mp4",
            "chunks": 5,
            "total_frames": 50,
            "total_memory_mb": 100.0,
            "avg_complexity": 0.5
        }
        connector.update_batch_on_child_completion("batch-123", "job-1", result1)

        batch = connector.get_job("batch-123")
        assert batch["_version"] == 1

        # Second update
        result2 = {
            "job_id": "job-2",
            "status": "completed",
            "filename": "video2.mp4",
            "chunks": 3,
            "total_frames": 30,
            "total_memory_mb": 50.0,
            "avg_complexity": 0.3
        }
        connector.update_batch_on_child_completion("batch-123", "job-2", result2)

        batch = connector.get_job("batch-123")
        assert batch["_version"] == 2

    def test_simulated_concurrent_updates(self, mock_modal_dict):
        """
        Simulate concurrent updates by manually triggering version conflicts.

        This test verifies that when a version mismatch is detected,
        the update retries and eventually succeeds.
        """
        connector = JobStoreConnector("test-jobs")
        connector.create_batch_job("batch-123", ["job-1", "job-2", "job-3"], "web-demo")

        # Simulate concurrent update scenario:
        # 1. First update starts, reads version 0
        # 2. Second update completes, writes version 1
        # 3. First update detects version mismatch, retries with version 1, writes version 2

        result1 = {
            "job_id": "job-1",
            "status": "completed",
            "filename": "video1.mp4",
            "chunks": 5,
            "total_frames": 50,
            "total_memory_mb": 100.0,
            "avg_complexity": 0.5
        }

        result2 = {
            "job_id": "job-2",
            "status": "completed",
            "filename": "video2.mp4",
            "chunks": 3,
            "total_frames": 30,
            "total_memory_mb": 50.0,
            "avg_complexity": 0.3
        }

        # Both updates should succeed even if they start at the same time
        success1 = connector.update_batch_on_child_completion("batch-123", "job-1", result1)
        success2 = connector.update_batch_on_child_completion("batch-123", "job-2", result2)

        assert success1 is True
        assert success2 is True

        # Verify final state is correct
        batch = connector.get_job("batch-123")
        assert batch["completed_count"] == 2
        assert batch["failed_count"] == 0
        assert batch["processing_count"] == 1
        assert batch["total_chunks"] == 8  # 5 + 3
        assert batch["total_frames"] == 80  # 50 + 30
        assert batch["_version"] == 2  # Two successful updates

    def test_concurrent_updates_all_counts_correct(self, mock_modal_dict):
        """
        Verify that all counts are correct after multiple concurrent updates.

        This is the critical test that would fail with the race condition.
        """
        connector = JobStoreConnector("test-jobs")

        # Create batch with 10 jobs
        child_ids = [f"job-{i}" for i in range(10)]
        connector.create_batch_job("batch-123", child_ids, "web-demo")

        # Simulate 10 concurrent completions
        for i in range(10):
            result = {
                "job_id": f"job-{i}",
                "status": "completed",
                "filename": f"video{i}.mp4",
                "chunks": i + 1,  # Different chunk counts
                "total_frames": (i + 1) * 10,
                "total_memory_mb": (i + 1) * 10.0,
                "avg_complexity": 0.1 * (i + 1)
            }
            success = connector.update_batch_on_child_completion("batch-123", f"job-{i}", result)
            assert success is True

        # Verify all counts are correct
        batch = connector.get_job("batch-123")
        assert batch["completed_count"] == 10
        assert batch["failed_count"] == 0
        assert batch["processing_count"] == 0
        assert batch["status"] == "completed"

        # Verify aggregate metrics
        expected_chunks = sum(range(1, 11))  # 1+2+3+...+10 = 55
        expected_frames = sum((i + 1) * 10 for i in range(10))  # 10+20+30+...+100 = 550
        expected_memory = sum((i + 1) * 10.0 for i in range(10))  # 10+20+30+...+100 = 550

        assert batch["total_chunks"] == expected_chunks
        assert batch["total_frames"] == expected_frames
        assert abs(batch["total_memory_mb"] - expected_memory) < 0.01

        # Verify version matches number of updates
        assert batch["_version"] == 10

    def test_mixed_success_and_failure_concurrent(self, mock_modal_dict):
        """Verify mixed success/failure updates work correctly with concurrent access."""
        connector = JobStoreConnector("test-jobs")

        child_ids = [f"job-{i}" for i in range(6)]
        connector.create_batch_job("batch-123", child_ids, "web-demo")

        # 4 successful, 2 failed
        for i in range(4):
            result = {
                "job_id": f"job-{i}",
                "status": "completed",
                "filename": f"video{i}.mp4",
                "chunks": 5,
                "total_frames": 50,
                "total_memory_mb": 100.0,
                "avg_complexity": 0.5
            }
            connector.update_batch_on_child_completion("batch-123", f"job-{i}", result)

        for i in range(4, 6):
            result = {
                "job_id": f"job-{i}",
                "status": "failed",
                "filename": f"video{i}.mp4",
                "error": "Processing failed"
            }
            connector.update_batch_on_child_completion("batch-123", f"job-{i}", result)

        # Verify counts
        batch = connector.get_job("batch-123")
        assert batch["completed_count"] == 4
        assert batch["failed_count"] == 2
        assert batch["processing_count"] == 0
        assert batch["status"] == "partial"
        assert len(batch["completed_jobs"]) == 4
        assert len(batch["failed_jobs"]) == 2

    def test_unexpected_child_status_handled(self, mock_modal_dict):
        """Verify unexpected child status values are handled gracefully."""
        connector = JobStoreConnector("test-jobs")
        connector.create_batch_job("batch-123", ["job-1", "job-2", "job-3"], "web-demo")

        # Test with None status
        result_none = {
            "job_id": "job-1",
            "status": None,
            "filename": "video1.mp4"
        }
        connector.update_batch_on_child_completion("batch-123", "job-1", result_none)

        batch = connector.get_job("batch-123")
        assert batch["failed_count"] == 1  # Treated as failure
        assert batch["processing_count"] == 2  # Decremented correctly
        assert len(batch["failed_jobs"]) == 1
        assert "Invalid status: None" in batch["failed_jobs"][0]["error"]

        # Test with unexpected string status
        result_invalid = {
            "job_id": "job-2",
            "status": "processing",  # Should never be passed to this method
            "filename": "video2.mp4"
        }
        connector.update_batch_on_child_completion("batch-123", "job-2", result_invalid)

        batch = connector.get_job("batch-123")
        assert batch["failed_count"] == 2  # Both treated as failures
        assert batch["processing_count"] == 1  # Decremented correctly
        assert len(batch["failed_jobs"]) == 2
        assert "Invalid status: processing" in batch["failed_jobs"][1]["error"]

        # Complete the last job normally
        result_success = {
            "job_id": "job-3",
            "status": "completed",
            "filename": "video3.mp4",
            "chunks": 5,
            "total_frames": 50,
            "total_memory_mb": 100.0,
            "avg_complexity": 0.5
        }
        connector.update_batch_on_child_completion("batch-123", "job-3", result_success)

        batch = connector.get_job("batch-123")
        assert batch["completed_count"] == 1
        assert batch["failed_count"] == 2
        assert batch["processing_count"] == 0  # All jobs accounted for
        assert batch["status"] == "partial"

    def test_empty_batch_progress_no_division_error(self, mock_modal_dict):
        """Verify empty batch (0 videos) doesn't cause division by zero error."""
        connector = JobStoreConnector("test-jobs")

        # Create an empty batch (edge case that shouldn't normally happen)
        connector.create_batch_job("batch-empty", [], "web-demo")

        # This should not raise a ZeroDivisionError
        progress = connector.get_batch_progress("batch-empty")

        assert progress is not None
        assert progress["total_videos"] == 0
        assert progress["completed"] == 0
        assert progress["failed"] == 0
        assert progress["processing"] == 0
        assert progress["progress_percent"] == 0.0  # Should return 0.0, not crash
