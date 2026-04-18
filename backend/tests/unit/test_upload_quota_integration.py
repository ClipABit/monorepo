"""
Tests for upload + quota flow integration.

Verifies user_id threading, quota checking, namespace override, and response enrichment.
"""

import pytest
from unittest.mock import MagicMock

from services.upload_handler import UploadHandler


@pytest.fixture
def mock_job_store():
    return MagicMock()


@pytest.fixture
def mock_spawn():
    return MagicMock()


@pytest.fixture
def handler(mock_job_store, mock_spawn):
    return UploadHandler(
        job_store=mock_job_store,
        process_video_spawn_fn=mock_spawn,
    )


class TestUploadPassesUserId:
    """Test that user_id flows from handler to spawn."""

    @pytest.mark.asyncio
    async def test_single_upload_passes_user_id_to_spawn(
        self, handler, mock_spawn, make_upload_file
    ):
        """user_id flows from handle_single_upload through to spawn."""
        file = make_upload_file()

        await handler.handle_single_upload(file, "ns", "auth0|user1")

        mock_spawn.assert_called_once()
        call_args = mock_spawn.call_args[0]
        assert call_args[5] == "auth0|user1"  # 6th positional arg is user_id

    @pytest.mark.asyncio
    async def test_single_upload_stores_user_id_in_job(
        self, handler, mock_job_store, make_upload_file
    ):
        """user_id is stored in the job metadata."""
        file = make_upload_file()

        await handler.handle_single_upload(file, "ns", "auth0|user1")

        mock_job_store.create_job.assert_called_once()
        job_data = mock_job_store.create_job.call_args[0][1]
        assert job_data["user_id"] == "auth0|user1"

    @pytest.mark.asyncio
    async def test_batch_upload_passes_user_id_to_spawn(
        self, handler, mock_spawn, make_upload_file
    ):
        """user_id flows through batch upload to each spawn call."""
        files = [make_upload_file("a.mp4"), make_upload_file("b.mp4")]

        await handler.handle_batch_upload(files, "ns", "auth0|batch_user")

        assert mock_spawn.call_count == 2
        for call_obj in mock_spawn.call_args_list:
            assert call_obj[0][5] == "auth0|batch_user"

    @pytest.mark.asyncio
    async def test_handle_upload_single_passes_user_id(
        self, handler, mock_spawn, make_upload_file
    ):
        """handle_upload with 1 file routes to single and passes user_id."""
        file = make_upload_file()

        await handler.handle_upload([file], "ns", "auth0|u1")

        mock_spawn.assert_called_once()
        assert mock_spawn.call_args[0][5] == "auth0|u1"

    @pytest.mark.asyncio
    async def test_handle_upload_batch_passes_user_id(
        self, handler, mock_spawn, make_upload_file
    ):
        """handle_upload with 2+ files routes to batch and passes user_id."""
        files = [make_upload_file("a.mp4"), make_upload_file("b.mp4")]

        await handler.handle_upload(files, "ns", "auth0|u2")

        assert mock_spawn.call_count == 2
        for call_obj in mock_spawn.call_args_list:
            assert call_obj[0][5] == "auth0|u2"


class TestUploadNullUserId:
    """Test backward compatibility when user_id is None."""

    @pytest.mark.asyncio
    async def test_single_upload_none_user_id(
        self, handler, mock_spawn, make_upload_file
    ):
        """Handles None user_id without error."""
        file = make_upload_file()

        await handler.handle_single_upload(file, "ns")

        mock_spawn.assert_called_once()
        assert mock_spawn.call_args[0][5] is None

    @pytest.mark.asyncio
    async def test_batch_upload_none_user_id(
        self, handler, mock_spawn, make_upload_file
    ):
        """Handles None user_id in batch without error."""
        files = [make_upload_file("a.mp4"), make_upload_file("b.mp4")]

        await handler.handle_batch_upload(files, "ns")

        for call_obj in mock_spawn.call_args_list:
            assert call_obj[0][5] is None


class TestUploadNamespaceFromJob:
    """Test that namespace is correctly stored in jobs."""

    @pytest.mark.asyncio
    async def test_single_upload_stores_namespace_in_job(
        self, handler, mock_job_store, make_upload_file
    ):
        """Namespace from parameter is stored in job metadata."""
        file = make_upload_file()

        await handler.handle_single_upload(file, "user_abc123", "auth0|u1")

        job_data = mock_job_store.create_job.call_args[0][1]
        assert job_data["namespace"] == "user_abc123"

    @pytest.mark.asyncio
    async def test_single_upload_passes_namespace_to_spawn(
        self, handler, mock_spawn, make_upload_file
    ):
        """Namespace is passed to the spawn function."""
        file = make_upload_file()

        await handler.handle_single_upload(file, "user_ns_xyz", "auth0|u1")

        call_args = mock_spawn.call_args[0]
        assert call_args[3] == "user_ns_xyz"  # 4th positional arg is namespace
