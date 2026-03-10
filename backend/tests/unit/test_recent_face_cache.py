from face_recognition.recent_faces_cache import RecentFacesCache
from face_recognition.recent_faces_cache import FaceData
import numpy as np

class TestRecentFacesCache:
    def test_add_and_get_face(self):
        cache = RecentFacesCache(max_size=2, confidente_threshold=0.8)
        face_data_1 = FaceData(face_id="face1", embedding=np.array([0.1, 0.2, 0.3]))
        face_data_2 = FaceData(face_id="face2", embedding=np.array([0.4, 0.5, 0.6]))

        cache.add_face("face1", face_data_1)
        cache.add_face("face2", face_data_2)

        assert cache.get_face("face1") == face_data_1
        assert cache.get_face("face2") == face_data_2

    def test_cache_eviction(self):
        cache = RecentFacesCache(max_size=2, confidente_threshold=0.8)
        face_data_1 = FaceData(face_id="face1", embedding=np.array([0.1, 0.2, 0.3]))
        face_data_2 = FaceData(face_id="face2", embedding=np.array([0.4, 0.5, 0.6]))
        face_data_3 = FaceData(face_id="face3", embedding=np.array([0.7, 0.8, 0.9]))

        cache.add_face("face1", face_data_1)
        cache.add_face("face2", face_data_2)
        cache.add_face("face3", face_data_3)  # This should evict face1

        assert cache.get_face("face1") is None
        assert cache.get_face("face2") == face_data_2
        assert cache.get_face("face3") == face_data_3

    def test_query_similar_face(self):
        cache = RecentFacesCache(max_size=3, confidente_threshold=0.8)
        face_data_1 = FaceData(face_id="face1", embedding=np.array([1.0, 0.0, 0.0]))
        face_data_2 = FaceData(face_id="face2", embedding=np.array([0.0, 1.0, 0.0]))
        face_data_3 = FaceData(face_id="face3", embedding=np.array([0.0, 0.0, 1.0]))

        cache.add_face("face1", face_data_1)
        cache.add_face("face2", face_data_2)
        cache.add_face("face3", face_data_3)
        query_embedding = np.array([0.9, 0.1, 0.0])
        result = cache.query_similar_face(query_embedding, similarity_funct="cosine")
        assert result is not None
        matched_face_data, similarity = result
        assert matched_face_data.face_id == "face1"
        assert similarity > 0.8

    def test_query_no_similar_face(self):
        cache = RecentFacesCache(max_size=2, confidente_threshold=0.95)
        face_data_1 = FaceData(face_id="face1", embedding=np.array([1.0, 0.0, 0.0]))
        face_data_2 = FaceData(face_id="face2", embedding=np.array([0.0, 1.0, 0.0]))

        cache.add_face("face1", face_data_1)
        cache.add_face("face2", face_data_2)
        query_embedding = np.array([0.5, 0.5, 0.0])
        result = cache.query_similar_face(query_embedding, similarity_funct="cosine")
        assert result is None

    def test_query_after_eviction(self):
        cache = RecentFacesCache(max_size=2, confidente_threshold=0.8)
        face_data_1 = FaceData(face_id="face1", embedding=np.array([1.0, 0.0, 0.0]))
        face_data_2 = FaceData(face_id="face2", embedding=np.array([0.0, 1.0, 0.0]))
        face_data_3 = FaceData(face_id="face3", embedding=np.array([0.0, 0.0, 1.0]))

        cache.add_face("face1", face_data_1)
        cache.add_face("face2", face_data_2)
        cache.add_face("face3", face_data_3)  # This should evict face1

        query_embedding = np.array([0.9, 0.1, 0.0])
        result = cache.query_similar_face(query_embedding, similarity_funct="cosine")
        assert result is None
        query_embedding_2 = np.array([0.0, 0.9, 0.1])
        result_2 = cache.query_similar_face(query_embedding_2, similarity_funct="cosine")
        assert result_2 is not None
        matched_face_data_2, similarity_2 = result_2
        assert matched_face_data_2.face_id == "face2"
        assert similarity_2 > 0.8

    def test_query_after_adding_existing_face(self):
        cache = RecentFacesCache(max_size=2, confidente_threshold=0.8)
        face_data_2 = FaceData(face_id="face2", embedding=np.array([0.0, 1.0, 0.0]))
        face_data_3 = FaceData(face_id="face3", embedding=np.array([0.0, 0.0, 1.0]))

        cache.add_face("face2", face_data_2)
        cache.add_face("face3", face_data_3)
        cache.add_face("face2", face_data_2)

        result_3 = cache.query_similar_face(np.array([0.0, 0.9, 0.1]), similarity_funct="cosine")
        assert result_3 is not None
        matched_face_data_3, similarity_3 = result_3
        assert matched_face_data_3.face_id == "face2"
        assert similarity_3 > 0.8

    def test_with_one_face_in_cache(self):
        cache = RecentFacesCache(max_size=2, confidente_threshold=0.8)
        face_data_1 = FaceData(face_id="face1", embedding=np.array([1.0, 0.0, 0.0]))

        cache.add_face("face1", face_data_1)

        query_embedding = np.array([0.9, 0.1, 0.0])
        result = cache.query_similar_face(query_embedding, similarity_funct="cosine")
        assert result is not None
        matched_face_data, similarity = result
        assert matched_face_data.face_id == "face1"
        assert similarity > 0.8