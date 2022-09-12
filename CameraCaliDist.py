import numpy as np
import cv2 as cv
import glob
import json
from datetime import datetime

#open JSON file
with open('camera.json', 'r') as json_file:
	camera_data = json.load(json_file)
#retrieve dist and mtx
dist = np.array(camera_data["dist"])
mtx = np.array(camera_data["mtx"])

images = glob.glob('calibrate*.png')
print(len(images), "images found")

assert len(images) > 0

#assume they all have the same dimensions
#get first image to determine the size of all the image
frame = cv.imread(images[0])
h, w = frame.shape[:2]

#calculate the specific camera matrix for h, w and region of interest
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (h, w), 0, (h, w))
x, y, w1, h1 = roi
yh1 = y + h1
xw1 = x + w1

#for each filename we found
for fname in images:
    #read the image from that file
	img = cv.imread(fname)

    #undistort the image into new image called dst
	dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    #cropping of the image according to the roi
	# dst = dst[y:yh1, x:xw1]

    #show the image that we just undistorted and save it
	cv.imshow('img', dst)
	cv.imwrite(f"remapped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dst)
	cv.waitKey(1500)

#credit
#https://betterprogramming.pub/how-to-calibrate-a-camera-using-python-and-opencv-23bab86ca194