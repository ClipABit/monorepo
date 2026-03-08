from database.cache.url_cache_connector import UrlCacheConnector, VIDEO_PAGE_TTL_SECONDS


class TestUrlCacheConnector:
    def test_get_page_uses_cache_until_expiry(self, mock_modal_dict, mocker):
        connector = UrlCacheConnector(environment="test")
        time_mock = mocker.patch("database.cache.url_cache_connector.time")
        time_mock.time.return_value = 1_000.0

        videos_payload = [
            {"file_name": "vid1.mp4", "presigned_url": "http://example/1"},
            {"file_name": "vid2.mp4", "presigned_url": "http://example/2"},
        ]

        connector.set_page(
            namespace="ns",
            page_token=None,
            page_size=2,
            videos=videos_payload,
            next_token="next-token",
        )

        cached_first = connector.get_page(namespace="ns", page_token=None, page_size=2)
        assert cached_first is not None
        assert cached_first["videos"] == videos_payload
        assert cached_first["next_token"] == "next-token"

        time_mock.time.return_value = 1_000.0 + VIDEO_PAGE_TTL_SECONDS - 5
        cached_second = connector.get_page(namespace="ns", page_token=None, page_size=2)
        assert cached_second is not None
        assert cached_second["videos"] == videos_payload

        time_mock.time.return_value = 1_000.0 + VIDEO_PAGE_TTL_SECONDS + 5
        expired_result = connector.get_page(namespace="ns", page_token=None, page_size=2)
        assert expired_result is None
        assert not mock_modal_dict

    def test_namespace_metadata_cache(self, mock_modal_dict, mocker):
        connector = UrlCacheConnector(environment="test")
        time_mock = mocker.patch("database.cache.url_cache_connector.time")
        time_mock.time.return_value = 2_000.0

        connector.set_namespace_metadata("ns", {"total_videos": 5})
        metadata_cached = connector.get_namespace_metadata("ns")
        assert metadata_cached == {"total_videos": 5, "cached_at": 2_000.0}

        time_mock.time.return_value = 2_000.0 + VIDEO_PAGE_TTL_SECONDS + 1
        metadata_expired = connector.get_namespace_metadata("ns")
        assert metadata_expired is None
        assert not mock_modal_dict

    def test_clear_namespace_removes_entries(self, mock_modal_dict, mocker):
        connector = UrlCacheConnector(environment="test")
        mocker.patch("database.cache.url_cache_connector.time").time.return_value = 3_000.0

        connector.set_page("ns", None, 10, [{"file_name": "vid.mp4"}], None)
        connector.set_namespace_metadata("ns", {"total_videos": 1})

        removed = connector.clear_namespace("ns")
        assert removed >= 2
        assert not mock_modal_dict
