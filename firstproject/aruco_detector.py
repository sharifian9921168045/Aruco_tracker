import cv2
import numpy as np
from cv2 import aruco
import mapper

# انواع مارکر ساپورت شده در opencv
ARUCO_DICT = {
	#  "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	#  "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	#  "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	#  "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	#  "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	#  "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	#  "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	#  "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	#  "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	#  "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	#  "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	#  "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	# "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	#  "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	 "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	 # "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	 # "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	 # "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	 # "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	 # "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	 # "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

#
def edge_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(blurred, 70, 215)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(canny, kernel)
    cv2.imshow('canny',dilated)
    return dilated



class Marker:
	def __init__(self,img,canny):
		self.Img = img
		self.blank = np.zeros(shape=self.Img.shape,dtype='uint8')
		self.canny = canny

	def crop_picture(self):

		#self.Img = cv2.resize(self.Img, (1300, 800))
		#cv2.imshow('img', image)  # resizing because opencv does not work well with bigger images
		orig = self.Img.copy()
		# gray = cv2.cvtColor(self.Img, cv2.COLOR_BGR2GRAY)  # RGB To Gray Scale
		# # (5,5) is the kernel size and 0 is sigma that determines the amount of blur
		# blurred = cv2.GaussianBlur(gray, (3, 3), 0)
		# edged = cv2.Canny(blurred, 70, 215)
		#cv2.imshow('canny' , edged)# 30 MinThreshold and 50 is the MaxThreshold
		# kernel = np.ones((3,3), np.uint8)
		# edged = cv2.dilate(edged,kernel,iterations=1)
		cv2.imshow('canny', self.canny)
		#cv2.imshow('edged',edged)
		# retrieve the contours as a list, with simple apprximation model
		contours, hierarchy = cv2.findContours(self.canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		#the loop extracts the boundary contours of the pagel
		for c in contours:
			p = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.1 * p, True)

			if len(approx) == 4:
				target = approx
				break
		approx = mapper.mapp(target)  # find endpoints of the sheet

		pts = np.float32([[0, 0], [1000, 0], [1000, 700], [0, 700]])  # map to 800*800 target window

		op = cv2.getPerspectiveTransform(approx, pts)  # get the top or bird eye view effect
		dst = cv2.warpPerspective(orig, op, (1000, 700))


		mask = np.zeros(orig.shape[:2], dtype=orig.dtype)
		#draw all contours larger than 20 on the mask
		for c in contours:
			if cv2.contourArea(c) > 1000:
				x, y, w, h = cv2.boundingRect(c)
				cv2.drawContours(mask, [c], 0, (255), -1)
				break

		# apply the mask to the original image
		result = cv2.bitwise_and(orig, orig, mask=mask)
		#
		# # show image
		#cv2.imshow("Result", result)
		#cv2.imshow("Im", orig)
		#print(dst.shape)
		#cv2.imshow("Scanned", dst)
		#return self.Img
		cv2.imwrite("perspective.jpg" , dst)
		return dst


	def calculate(self,x_in , y_in ,arz , artfa , topL , topR , buttomL):

		arz_pixel = int(topR[0] - topL[0])
		artfa_pixel =int(buttomL[1] - topL[1])
		#print(f'arz_pixel :{arz_pixel}')
		#print(f'artfa_pixel :{artfa_pixel}')
		arz_pixel_per_cm = arz/ arz_pixel
		artfa_pixel_per_cm = artfa/ artfa_pixel
		dist_x = x_in - topL[0]
		dist_y = y_in - topL[1]
		# "{:.2f}".format(dist_x * arz_pixel_per_cm)
		return "{:.3f}".format(dist_x * arz_pixel_per_cm) , "{:.3f}".format(dist_y * artfa_pixel_per_cm )


	def detect_marker(self,img): # تابع برای تشخیص مارکر
		aruco_name, aruco_dict = ("DICT_7X7_250",cv2.aruco.DICT_7X7_50)  # مقدار دهی برای مارکر مشخص
		aruco_dict = aruco.getPredefinedDictionary(aruco_dict) #
		aruco_params = aruco.DetectorParameters()
		corners, ids, rejected = aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
		int_corners = np.intp(corners)
		cv2.polylines(img, int_corners, True, (0, 255, 0), thickness=2)  # bold marker
		cv2.polylines(self.blank, int_corners, True, (255, 255, 255), thickness=1)  # detect delimeter of marker
		#self.Img = cv2.resize(self.Img,(1000,600),interpolation=cv2.INTER_AREA)
		#cv2.imshow('blank', self.blank)
		cv2.imshow('image', img)
		return img,self.blank

	#
	def marker_location(self):
		gray = cv2.cvtColor(self.blank,cv2.COLOR_BGR2GRAY)
		contours, hierarchies = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		c = 0
		li = []
		for i in contours:
			M = cv2.moments(i)
			if M['m00'] != 0:
				c+=1
				cx = int(M['m10'] / M['m00'])
				cy = int(M['m01'] / M['m00'])
				cv2.circle(gray,center=(cx,cy),radius=2,color=(255,255,255),thickness=-1)
				if c%2==1:
					li.append([cx,cy])
		# cv2.imshow('centers',gray)
		# cv2.moveWindow('centers', 0, 0)
		return np.array(li).squeeze()
	def tracker(self , image , location):
		cv2.circle(image ,(location[0],location[1]) ,2 ,  (0 , 0 , 255) , -1 )


#پیداکردن طول و عرض و ارتفاع

def pose_estimation(frame , aruco_dict_type , matrix_coeff , distortion):
	for arucoName, arucoDict in ARUCO_DICT.items():
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		aruco_dict = cv2.aruco.getPredefinedDictionary(arucoDict)
		parameters = cv2.aruco.DetectorParameters()
		corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters) #cameraMatrix=matrix_coeff, discoeff=distortion)


	if len(corners) >  0:
		for i in range(0 , len(ids)):
			rvec , tvec , markerpoint = cv2.aruco.estimatePoseSingleMarkers(corners[i] , 0.02 , matrix_coeff , distortion)
			cv2.aruco.drawDetectedMarkers(frame , corners)
			cv2.drawFrameAxes(frame , matrix_coeff , distortion , rvec , tvec ,  0.01)






