import numpy as np
import cv2 as cv
import glob
import json
from datetime import datetime

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

rows = 5
columns = 7
objp = np.zeros((rows*columns,3), np.float32)
objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)

#create arrays to store the object points and images from current directory
objpoints = []
imgpoints = []

#use glob to find all the files starting with 'calibrate' and .png extension
images = glob.glob('calibrate*.png')
#load the images
print(len(images), "images found")

for fname in images:
    #convert the images into grayscale
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #chessboard flags guide the algorithm that searches for chessboard corners.
    #could also set the flags to None
    chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv.findChessboardCorners(gray, (columns,rows), chessboard_flags)

    if ret == True:
        #add the corners to objpoints array
        objpoints.append(objp)
        #refine the corner locations
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        #add the empty 3D object points to imgpoints array
        imgpoints.append(corners)

        #draw the corners we found in the image and output these images
        cv.drawChessboardCorners(img, (columns,rows), corners2, ret)
        cv.imshow('img', img)
        cv.imwrite(f"chessboard_corners_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", img)
        cv.waitKey(1500)

#calibrate the camera
#mtx = camera matrix
#dist = distance coefficients
#rvecs = rotation
#tvecs = translate vectors
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#return all the saved variables in JSON file
camera = {}

#given some variables are NumPy arrays, run the encoder to save them as lists
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

for variable in ['ret', 'mtx', 'dist', 'rvecs', 'tvecs']:
    camera[variable] = eval(variable)

with open("camera.json", 'w') as f:
    json.dump(camera, f, indent=4, cls=NumpyEncoder)

cv.destroyAllWindows()

#credit
#https://betterprogramming.pub/how-to-calibrate-a-camera-using-python-and-opencv-23bab86ca194