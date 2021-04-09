import cv2
import numpy as np

# This is a contrast based histogram equalization, using this we can have a better contrast between our grayscale values
# The equalization is done by blocks, that is every 8x8 block is equalized, but that would increase noise if there is noise
# to avoid it we use the clipLimit, so if the histogram bin is higher than the clipLimit we distribute it's value uniformly
# to the other bins before equalization.
def CLAHE(frame, clipLimit = 2.0, tileGridSize = (8,8)):
	clahe = cv2.createCLAHE(clipLimit, tileGridSize)
	CLAHEFrame = clahe.apply(frame)
	return CLAHEFrame

# The blur function is to get a better detail quality each blur has a purpose, median usually is good with cleaning up
# noise while gaussian is better at preserving the edges.
def blur(EqFrame, kernelSize = 7, blurType = 0):
	if (blurType == 0):
		blurredFrame = cv2.GaussianBlur(EqFrame,(kernelSize, kernelSize), 0)
	elif (blurType == 1):
		blurredFrame = cv2.medianBlur(EqFrame, kernelSize)
	return blurredFrame

# Just flips the image horizontaly and converts to Grayscale
def initialAdjustments(frame):
	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hFlipGrayFrame = cv2.flip(grayFrame, 1)
	return hFlipGrayFrame

# When we normalize the image we are "stretching" the histogram and correction some of the bad light in it,
# then we procced to the tile based contrast histogram equalization and finally the entire image histogram equalization
# so that our bad light conditions as well as the contrast are better and the grayscale image has more detail
def correctLighting(hFlipGrayFrame):
	normalizedFrame = cv2.normalize(hFlipGrayFrame, None, 0, 255, cv2.NORM_MINMAX)
	CLAHEFrame = CLAHE(normalizedFrame)
	EqFrame = cv2.equalizeHist(CLAHEFrame)
	return EqFrame

# In the end we use a blur to both clean some of the noise we might have as well as smooth our edges so its easier to detect
# and finally threshold the frame to better visualize the edges
def preProcess(frame):
	adjustedFrame = initialAdjustments(frame)
	correctedFrame = correctLighting(adjustedFrame)
	clearerFrame = blur(correctedFrame)
	thresholdFrame = cv2.adaptiveThreshold(clearerFrame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
             cv2.THRESH_BINARY,11,2)
	return thresholdFrame 