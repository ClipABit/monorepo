from firebase_admin import firestore

class FaceAppearanceRepository:
    """
    Repository for storing and retrieving face appearance data (i.e. which face appear in which clip chunk) in firebase
    """
    def __init__(self, db: firestore.Client):
        self._db = db

    def _faces_appearances_ref(self, user_id: str):
        return (
            self._db
            .collection("users").
            document(user_id)
            .collection("face_appearances")
        )

    def set_face_appearance(self, user_id: str, face_id: str, video_chunk_id: str) -> bool:
        """Store face appearance data for a given user, face ID, and video chunk ID. Creates or overwrites the document.

        Args:
            user_id (str): The ID of the user.
            face_id (str): The ID of the face.
            video_chunk_id (str): The ID of the video chunk.
        Returns:
            bool: True if storage was successful, False otherwise.
        """
        try:
            face_appearance_ref = self._faces_appearances_ref(user_id).document(f"{face_id}_{video_chunk_id}")
            face_appearance_ref.set({
                "face_id": face_id,
                "video_chunk_id": video_chunk_id
            })
            return True
        except Exception as e:
            print(f"Error storing face appearance data: {e}")
            return False
        
    def get_faces_for_chunk(self, user_id: str, video_chunk_id: str) -> dict | None:
        """
        Retrieve face appeared in a given user and video chunk ID.
        Args:
            user_id (str): The ID of the user.
            video_chunk_id (str): The ID of the video chunk.
        Returns:
            dict | None: The face appearance data dictionary if found, None otherwise.
        """
        try:
            face_appearances_ref = self._faces_appearances_ref(user_id)
            query = face_appearances_ref.where("video_chunk_id", "==", video_chunk_id)
            docs = query.stream()
            appearances = {}
            for doc in docs:
                appearances[doc.id] = doc.to_dict()
            return appearances
        except Exception as e:
            print(f"Error retrieving face appearance data: {e}")
            return None
        
    def get_chunks_for_face(self, user_id: str, face_id: str) -> dict | None:
        """
        Retrieve video chunks where a given face appears for a user.
        Args:
            user_id (str): The ID of the user.
            face_id (str): The ID of the face.
        Returns:
            dict | None: The face appearance data dictionary if found, None otherwise.
        """
        try:
            face_appearances_ref = self._faces_appearances_ref(user_id)
            query = face_appearances_ref.where("face_id", "==", face_id)
            docs = query.stream()
            appearances = {}
            for doc in docs:
                appearances[doc.id] = doc.to_dict()
            return appearances
        except Exception as e:
            print(f"Error retrieving face appearance data: {e}")
            return None

    def delete_face_appearance(self, user_id: str, face_id: str, video_chunk_id: str) -> bool:
        """Delete face appearance data for a given user, face ID, and video chunk ID.

        Args:
            user_id (str): The ID of the user.
            face_id (str): The ID of the face.
            video_chunk_id (str): The ID of the video chunk.
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            face_appearance_ref = self._faces_appearances_ref(user_id).document(f"{face_id}_{video_chunk_id}")
            face_appearance_ref.delete()
            return True
        except Exception as e:
            print(f"Error deleting face appearance data: {e}")
            return False
        
    def update_face_appearance(self, user_id: str, face_id: str, video_chunk_id: str, update_data: dict) -> bool:
        """Update face appearance data for a given user, face ID, and video chunk ID.

        Args:
            user_id (str): The ID of the user.
            face_id (str): The ID of the face.
            video_chunk_id (str): The ID of the video chunk.
            update_data (dict): The data to update.
        Returns:
            bool: True if update was successful, False otherwise.
        """
        try:
            face_appearance_ref = self._faces_appearances_ref(user_id).document(f"{face_id}_{video_chunk_id}")
            face_appearance_ref.update(update_data)
            return True
        except Exception as e:
            print(f"Error updating face appearance data: {e}")
            return False