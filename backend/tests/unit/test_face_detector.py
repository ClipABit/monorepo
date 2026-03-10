from face_recognition.face_detector import FaceDetector

class TestBasicFaceDetector:
    def test_initialization(self):
        detector = FaceDetector(
            detector_backend="mtcnn",
            embedding_model_name="ArcFace",
            enforce_detection=True,
            align=True
        )
        assert detector.detector_backend == "mtcnn"
        assert detector.embedding_model_name == "ArcFace"
        assert detector.enforce_detection is True
        assert detector.align is True

    def test_detect_and_embed_no_faces(self, sample_image_no_face):
        detector = FaceDetector()
        faces = detector.detect_and_embed(sample_image_no_face)
        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_detect_and_embed_with_faces(self, sample_image_10x3_faces):
        detector = FaceDetector()
        faces = detector.detect_and_embed(sample_image_10x3_faces)
        assert isinstance(faces, list)
        assert len(faces) == 30  # Expecting 30 faces in the sample image
        for face in faces:
            assert hasattr(face, 'embedding')
            assert hasattr(face, 'face_image')