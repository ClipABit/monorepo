from firebase_admin import firestore

class FaceMetadataRepository:
    """
    Repository for storing and retrieving face metadata in Firestore.
    Each user's face metadata is stored in a subcollection "faces" under their user document.
    Metadata include the following:
    - face_id (str): Unique identifier for the face, used as the document ID.
    - image_url (str): URL to the stored face image.
    - given_name (str): Given name associated with the face.
    """
    def __init__(self, db: firestore.Client):
        self._db = db

    def _faces_ref(self, user_id: str):
        return (
            self._db
            .collection("users")
            .document(user_id)
            .collection("faces")
        )
    
    def set_face_metadata(self, user_id: str, face_id: str, metadata: dict) -> bool:
        """Store face metadata for a given user and face ID. Creates or overwrites the document.

        Args:
            user_id (str): The ID of the user.
            face_id (str): The ID of the face.
            metadata (dict): The face metadata to store.
        Returns:
            bool: True if storage was successful, False otherwise.
        """        
        try:
            face_ref = self._faces_ref(user_id).document(face_id)
            face_ref.set(metadata)
            return True
        except Exception as e:
            print(f"Error storing face metadata: {e}")
            return False
        
    def get_face_metadata(self, user_id: str, face_id: str) -> dict | None:
        """Retrieve face metadata for a given user and face ID.

        Args:
            user_id (str): The ID of the user.
            face_id (str): The ID of the face.
        Returns:
            dict | None: The face metadata dictionary if found, None otherwise.
        """
        try:
            face_ref = self._faces_ref(user_id).document(face_id)
            doc = face_ref.get()
            if doc.exists:
                return doc.to_dict()
            else:
                print(f"Face metadata not found for user_id={user_id}, face_id={face_id}")
                return None
        except Exception as e:
            print(f"Error retrieving face metadata: {e}")
            return None
        
    def delete_face_metadata(self, user_id: str, face_id: str) -> bool:
        """Delete face metadata for a given user and face ID.

        Args:
            user_id (str): The ID of the user.
            face_id (str): The ID of the face.
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            face_ref = self._faces_ref(user_id).document(face_id)
            face_ref.delete()
            return True
        except Exception as e:
            print(f"Error deleting face metadata: {e}")
            return False
    
    def list_face_metadata(self, user_id: str) -> list[dict]:
        """List all face metadata for a given user.

        Args:
            user_id (str): The ID of the user.
        Returns:
            list[dict]: A list of face metadata dictionaries.
        """
        try:
            faces_ref = self._faces_ref(user_id)
            docs = faces_ref.stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            print(f"Error listing face metadata: {e}")
            return []
        
    def update_face_metadata(self, user_id: str, face_id: str, update_data: dict) -> bool:
        """Update face metadata for a given user and face ID for an existing document.

        Args:
            user_id (str): The ID of the user.
            face_id (str): The ID of the face.
            update_data (dict): The fields to update in the face metadata.
        Returns:
            bool: True if update was successful, False otherwise.
        """
        try:
            face_ref = self._faces_ref(user_id).document(face_id)
            face_ref.update(update_data)
            return True
        except Exception as e:
            print(f"Error updating face metadata: {e}")
            return False
    