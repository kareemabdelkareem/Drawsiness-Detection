# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import time
from playsound import playsound

file = open("output.txt", "w")
def eye_aspect_ratio(eye):
	A = dist.euclidean([eye[1].x,eye[1].y], [eye[5].x,eye[5].y])
	B = dist.euclidean([eye[2].x,eye[2].y], [eye[4].x,eye[4].y])
	C = dist.euclidean([eye[0].x,eye[0].y], [eye[3].x,eye[3].y])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear
	
def take_frames(cap,detector,predictor,state,ear_arr):
	for i in range(50):
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		faces = detector(gray)
		
		for face in faces:	
			x1 = face.left()
			y1 = face.top()
			x2 = face.right()
			y2 = face.bottom()
			#cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
			landmarks = predictor(gray, face)
			eye1=[]
			eye2=[]
			for n in range(36, 48):
				x = landmarks.part(n).x
				y = landmarks.part(n).y
				if n < 42: 
					eye1.append(landmarks.part(n))
				else :
					eye2.append(landmarks.part(n))
					
				#cv2.circle(frame, (x, y), 4, (255, 0, 0), -1) 
			right=eye_aspect_ratio(eye1)
			left=eye_aspect_ratio(eye2)
			
			ear=(right+left)/2.0
			ear_arr.append(ear)
			if len(ear_arr)>20:
				ear_arr.pop(0)
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 35),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, state, (100, 35),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)	
			cv2.putText(frame, "avg_ear: {:.2f}".format(np.mean(ear_arr)), (15, 55),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
		cv2.imshow("frame",frame)
		key=cv2.waitKey(1)
		if key ==27:
			break
			
	return np.mean(ear_arr)
	

def main():
	avg_ear=.21
	open=.30
	close=.15
	ear_arr=[]

	#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	cap = cv2.VideoCapture(0)
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	# play welcome voice meassage to indicate the system is running
	playsound('New folder/welcome1.mp3')
	time.sleep(.5)
	# play voice message that indicate the driver to open his eyes
	playsound('New folder/arab_openyoureye.mp3')
	time.sleep(.5)
	#take frame return the average EAR  for the open eye
	open=take_frames(cap,detector,predictor,"please open your eyes",ear_arr)
	#playsound of message indicate the driver to close his eyes 
	playsound('New folder/arabcloseypureyes.mp3')
	time.sleep(.5)
	# close is the average EAR of the closed eyes
	close=take_frames(cap,detector,predictor,"please close your eyes",ear_arr)
	# voice message to indicate the driver that the threshold is calibrated 
	playsound('New folder/arab_done.mp3')
	
	
	#file.write(str((open+close)/2.0))
	file.write(str((open+close)*.55))
	

	file.close()

	
	print("open= {:.2f} \nclose= {:.2f} \naverage aspect = {:.2f}".format(open,close,(open+close)*0.55))

if __name__ == '__main__' :
	main()
	
	
	
	
	
	
	"""while True:
		
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		faces = detector(gray)
		
		for face in faces:
		
				
			x1 = face.left()
			y1 = face.top()
			x2 = face.right()
			y2 = face.bottom()
			#cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
			landmarks = predictor(gray, face)
			eye1=[]
			eye2=[]
			for n in range(36, 48):
				x = landmarks.part(n).x
				y = landmarks.part(n).y
				if n < 42: 
					eye1.append(landmarks.part(n))
				else :
					eye2.append(landmarks.part(n))
					
				cv2.circle(frame, (x, y), 4, (255, 0, 0), -1) 
			right=eye_aspect_ratio(eye1)
			left=eye_aspect_ratio(eye2)
			
			ear=(right+left)/2.0
			ear_arr.append(ear)
			if len(ear_arr)>5:
				ear_arr.pop(0)
				
			
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 35),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "avg_ear: {:.2f}".format(avg_ear), (100, 35),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
			
			   
		        
		#cv2.imshow("frame",frame)           
		key=cv2.waitKey(1)
		if key == 99:
			open=np.mean(ear_arr)
		if key == 110:
			close=np.mean(ear_arr)
			avg_ear=(open+close)/2.0
		cv2.putText(frame, "open=: {:.2f}, close={:.2f}".format(open,close), (15, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)	
		cv2.imshow("frame",frame)
		if key == 27:
			break
			"""
	
	
	