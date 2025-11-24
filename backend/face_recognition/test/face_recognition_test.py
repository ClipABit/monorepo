import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from face_recognition import FaceRepository


def main():
	fr = FaceRepository(
		pinecone_api_key=os.getenv("PINECONE_API_KEY"),
		index_name="face-index"
    )
	faces = fr.add_images(namespace="test", chunk_id="chunk_0", img_lst=["photo1.jpg"])
	print(faces)

if __name__ == "__main__":
	main()
