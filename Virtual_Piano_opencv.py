#OpenCv test
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import pygame


def segment(image):
        #converting color image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ##Creating a mask to  return the pixels of the detection
        mask_black = cv2.inRange(gray_image, 0, 25)
        print(mask_black.shape)

        #finding Contours
        _,contours,hierarchy = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print("contour length",len(contours))
        if len(contours) > 0:
                # Find the index of the largest contour
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cnt=contours[max_index]

                #Drawing bounding box
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(mask_black,(x,y),(x+w,y+h),(255),2)

                pygame.mixer.init()#Initilazing Audio player

                #Finding the segment
                imshape = image.shape
                if (x < imshape[1]//7):
                        cv2.putText(image,"Sa", (0+15,imshape[0]//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                        # show the frame and playing audio
                        cv2.imshow("Frame", image)
                        cv2.waitKey(1)
                        pygame.mixer.music.load("SA.wav")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() == True:
                                continue
                        
                elif(x > imshape[1]//7 and x < (imshape[1]*2)//7):
                        cv2.putText(image,"Re", (imshape[1]//7+15,imshape[0]//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                        cv2.imshow("Frame", image)
                        cv2.waitKey(1)
                        pygame.mixer.music.load("RE.wav")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() == True:
                                continue
                        
                elif(x > (imshape[1]*2)//7 and x < (imshape[1]*3)//7):
                        cv2.putText(image,"Ga", ((imshape[1]*2)//7+15,imshape[0]//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                        cv2.imshow("Frame", image)
                        cv2.waitKey(1)
                        pygame.mixer.music.load("GA.wav")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() == True:
                                continue
                        
                elif(x > (imshape[1]*3)//7 and x < (imshape[1]*4)//7):
                        cv2.putText(image,"Ma", ((imshape[1]*3)//7+15,imshape[0]//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                        cv2.imshow("Frame", image)
                        cv2.waitKey(1)
                        pygame.mixer.music.load("MA.wav")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() == True:
                                continue
                        
                elif(x > (imshape[1]*4)//7 and x < (imshape[1]*5)//7):
                        cv2.putText(image,"Pa", ((imshape[1]*4)//7+15,imshape[0]//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                        cv2.imshow("Frame", image)
                        cv2.waitKey(1)
                        pygame.mixer.music.load("PA.wav")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() == True:
                                continue
                        
                elif(x > (imshape[1]*5)//7 and x < (imshape[1]*6)//7):
                        cv2.putText(image,"Dha", ((imshape[1]*5)//7+10,imshape[0]//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                        cv2.imshow("Frame", image)
                        cv2.waitKey(1)
                        pygame.mixer.music.load("DHA.wav")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() == True:
                                continue
                        
                elif(x > (imshape[1]*6)//7 and x < imshape[1]):
                        cv2.putText(image,"Ni", ((imshape[1]*6)//7+15,imshape[0]//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                        cv2.imshow("Frame", image)
                        cv2.waitKey(1)
                        pygame.mixer.music.load("NI.wav")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() == True:
                                continue

        
        return mask_black
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (656, 304)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(656, 304))
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array

	#Virtually dividing the farme into 7
	color = [0,255,0]
	thickness = 3
	imshape = image.shape
	cv2.line(image,(imshape[1]//7,0),(imshape[1]//7,imshape[0]),color,thickness)
	cv2.line(image,(((imshape[1])*2)//7,0),(((imshape[1])*2)//7,imshape[0]),color,thickness)
	cv2.line(image,(((imshape[1])*3)//7,0),(((imshape[1])*3)//7,imshape[0]),color,thickness)
	cv2.line(image,(((imshape[1])*4)//7,0),(((imshape[1])*4)//7,imshape[0]),color,thickness)
	cv2.line(image,(((imshape[1])*5)//7,0),(((imshape[1])*5)//7,imshape[0]),color,thickness)
	cv2.line(image,(((imshape[1])*6)//7,0),(((imshape[1])*6)//7,imshape[0]),color,thickness)

	#calling segmentation function
	seg_image = segment(image)
	#cv2.imshow("Segmentation", seg_image)
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if cv2.waitKey(1)&0xFF==ord('q'):
		cv2.destroyWindow("Frame")
		cv2.destroyWindow("Segmentation")
		break
