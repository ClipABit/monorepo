from cache.video_cache import VideoCache, VIDEO_PAGE_TTL_SECONDS

class TestVideoCachePagination:
    def test_get_page_respects_ttl(self, mock_modal_dict, mocker):
        time_mock = mocker.patch("cache.video_cache.time")
        time_mock.time.return_value = 1_000.0

        cache = VideoCache(environment="dev")
        cache.set_page(
            namespace="ns",
            page_token=None,
            page_size=20,
            videos=[{"file_name": "vid.mp4"}],
            next_token="token",
        )

        # Within TTL -> entry returned
        time_mock.time.return_value = 1_000.0 + VIDEO_PAGE_TTL_SECONDS - 10
        entry = cache.get_page("ns", None, 20)
        assert entry is not None
        assert entry["videos"][0]["file_name"] == "vid.mp4"

        # Beyond TTL -> entry evicted and not returned
        time_mock.time.return_value = 1_000.0 + VIDEO_PAGE_TTL_SECONDS + 1
        expired_entry = cache.get_page("ns", None, 20)
        assert expired_entry is None
        assert not mock_modal_dict  # key removed

    def test_namespace_metadata_respects_ttl(self, mock_modal_dict, mocker):
        time_mock = mocker.patch("cache.video_cache.time")
        time_mock.time.return_value = 2_000.0

        cache = VideoCache(environment="dev")
        cache.set_namespace_metadata("ns", {"total_videos": 5})

        time_mock.time.return_value = 2_000.0 + VIDEO_PAGE_TTL_SECONDS - 10
        metadata = cache.get_namespace_metadata("ns")
        assert metadata is not None
        assert metadata["total_videos"] == 5

        time_mock.time.return_value = 2_000.0 + VIDEO_PAGE_TTL_SECONDS + 1
        expired_metadata = cache.get_namespace_metadata("ns")
        assert expired_metadata is None
        assert not mock_modal_dict
