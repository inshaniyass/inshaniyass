import numpy as np
import h5py
import dlib
import cv2

from PIL import Image
import imutils
from imutils import face_utils

SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
IMAGE_SIZE = 50

f = h5py.File('images.h5','r+')

images = np.array(f['data'])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

faceFeatures = []
faces = []
for image in images:

	

	image = np.uint8(image * 255)
	rects = detector(image, 0)

	for rect in rects:
		face = image[rect.top():rect.bottom(), rect.left():rect.right()].copy() 
		face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
		shape = predictor(face, dlib.rectangle(left=0, top=0, right=IMAGE_SIZE, bottom=IMAGE_SIZE))
		poi = []
		for i in range(17,68):
			poi.append([shape.part(i).x, shape.part(i).y])
		faceFeatures.append(poi)
		
	if (len(rects) == 0):
		print("Error: face not recognized")
		
		cv2.imshow("Image", image)
		cv2.waitKey(0)


faceFeatures = np.stack(faceFeatures)

print(faceFeatures.shape)
if ("faceFeatures" in f):
	del f['faceFeatures']
if ("faces" in f):
	del f['faces']

y_dset = f.create_dataset('faceFeatures', faceFeatures.shape, dtype='i')
y_dset[:] = faceFeatures



f.close()
