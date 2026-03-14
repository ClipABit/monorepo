import os
import sys
from PIL import Image
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from face_recognition import *


def main():
	face_detector = FaceDetector(detector_backend="retinaface")
	# img_path = "test_img_2faces.png"
	img_path = "/Users/yifanzhang/workspace/ClipABit/monorepo/backend/face_recognition/test/frame_images/saved_image_19.png"
	# img = Image.open(img_path)
	# arr = np.array(img)
	# # arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
	# faces = face_detector.detect_and_embed(img=arr)
	# faces = face_detector.detect_and_embed(img_path)
	faces = face_detector.detect_and_embed(np.load("/Users/yifanzhang/workspace/ClipABit/monorepo/backend/face_recognition/test/frame_images/img_array19.npy"))
	
	arr = np.load("/Users/yifanzhang/workspace/ClipABit/monorepo/backend/face_recognition/test/frame_images/img_array19.npy")
	# print("Loaded info:", type(arr), arr.shape, arr.dtype, np.min(arr), np.max(arr))
	print("Loaded top-left pixel:", arr[0,0])

	# print(arr)


	print(faces)

if __name__ == "__main__":
	main()
