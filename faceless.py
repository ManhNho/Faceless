 # -*- coding: utf-8 -*-
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2
from tensorflow_addons.layers import SpatialPyramidPooling2D
from efficientnet.tfkeras import EfficientNetB0
import argparse
from imutils.video import VideoStream

def banner():
	print(''' 
***++++++===================++++***
**++++=====-=+++++++==---=====+++++
**++=====-=*##%%%%%%%**+======+++++
**+++====+##*+****##**#%*====++++++
**++++===#*-::::::--=++*#+===++++++
***+++===#=::::::::-==+**+===+++++*
**++++===*=::::::::-==++*+===+++++*
**++=====+=::::::::--=++*+====++++*
**+++======:::::::---=++++====+++**
**+++++====---::::--=++=+=====+++**
***+++=++++----::---=++========++++
****+++++++==------==++==+++===++++
#****++++++=-===--=+***++++++=++++*
#*******+++..:-=+*###*==+++++++++**
###*******#   .:-+++-:-=#*++++++++*
##########%   .-+++=..:#%%%%#******
#%##%%%%%%%-.=*%#%%+=:-%%%%%%%%%###
#####%%%%%%=::.-%%=-..-%%%%%%%%%%%@
#####%###%%+ ..#%%%-..+%%%%%%%%%%%@
============ .:+==+=. =++++++++++++

Faceless v1.0 - ManhNho
		''')

def dectect_with_files():
	thresh = 0.5
	# Running the face detector model
	net = cv2.dnn.readNetFromCaffe('DNN_face_detector/deploy.prototxt',
    	                           'DNN_face_detector/res10_300x300_ssd_iter_140000.caffemodel')
	# Load the trained deepfake deep learning model
	model = load_model('model.h5',
					   custom_objects={'SpatialPyramidPooling2D': SpatialPyramidPooling2D})
	# Load Label Encoder
	le = pickle.loads(open('le.pickle', "rb").read())
	#  Read video
	video = input('Please input your video path:' )
	cap = cv2.VideoCapture(video)
	fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
	out = cv2.VideoWriter("video.avi", fourcc, 30, (1366, 768))
	while (cap.isOpened()):
		# read frame one by one
		_,frame = cap.read()
		frame = cv2.resize(frame, (1366, 768))
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > thresh:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				startX = max(0, startX)
				startY = max(0, startY)
				endX = min(w, endX)
				endY = min(h, endY)
				face = frame[startY:endY, startX:endX]
				try:
					face = cv2.resize(face, (224,224))
				except Exception as e:
					continue
				face = face.astype("float") / 255.0
				face = img_to_array(face)
				face = np.expand_dims(face, axis=0)
				preds = model.predict(face)[0]
				j = np.argmax(preds)
				label = le.classes_[j]
				label = "{}: {:.4f}".format(label, preds[j])
				if (j==0):
					cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
				else:
					cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0,  255,0), 2)
		out.write(frame)
		cv2.namedWindow("Detect Face", cv2.WINDOW_NORMAL)
		cv2.imshow("Detect Face", frame)
		if cv2.waitKey(1) == 27:
			break
	cap.release()
	out.release()
	cv2.destroyAllWindows()

def dectect_with_webcam():
	thresh = 0.5
	# Running the face detector model
	net = cv2.dnn.readNetFromCaffe('DNN_face_detector/deploy.prototxt',
    	                           'DNN_face_detector/res10_300x300_ssd_iter_140000.caffemodel')
	# Load the trained deepfake deep learning model
	model = load_model('model.h5',
					   custom_objects={'SpatialPyramidPooling2D': SpatialPyramidPooling2D})
	# Load Label Encoder
	le = pickle.loads(open('le.pickle', "rb").read())
	print("Lauching WEBCAM...")
	vs = VideoStream(0).start()
	fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
	out = cv2.VideoWriter("video.avi", fourcc, 32, (320, 240))
	while True:
		frame = vs.read()
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > thresh:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				startX = max(0, startX)
				startY = max(0, startY)
				endX = min(w, endX)
				endY = min(h, endY)
				face = frame[startY:endY, startX:endX]
				try:
					face = cv2.resize(face, (224,224))
				except Exception as e:
					continue
				face = face.astype("float") / 255.0
				face = img_to_array(face)
				face = np.expand_dims(face, axis=0)
				preds = model.predict(face)[0]
				j = np.argmax(preds)
				label = le.classes_[j]
				label = "{}: {:.4f}".format(label, preds[j])
				if (j==0):
					cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
				else:
					cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0,  255,0), 2)
		out.write(frame)
		cv2.imshow("Frame", frame)
		cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
		if cv2.waitKey(1) == 27:
			break
	out.release()
	cv2.destroyAllWindows()
	vs.stop()	

def main():
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	banner()
	parser = argparse.ArgumentParser() # Create ArgumentParser object
	parser.add_argument('-m','--method',required=True, type=int, choices=[1, 2], help='Select method for detect deepfake, \
		Select 1: Define video\'s local path \
		Select 2: using Webcam')
	args = parser.parse_args() # Return opts and args
	if args.method == 1:
		dectect_with_files()
	else:
		dectect_with_webcam()

if __name__ == '__main__':
	main()