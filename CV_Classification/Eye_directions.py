# What I did here was a very simple calibration system to get the gaze average infos when looking at the center position, from those infos try to classify where the person is looking at.
# After a calibration module is done this module will need to be upgraded and is likely to work consistently if we have a margin of a certain amount of pixels, to be more 
# precise.
# The classification works by getting the gaze average coordinates as well as the contour area after 5 frames, after that we take the absolute value of the difference from
# the center average position and the current gaze average position, then we compare those differences to know if the movement is bigger in the x axis or y axis
# and at the end by comparing the values in relation to the average center position we can decide where the person is looking at
# The calibration system works in a frame limit base, meaning that after 80 frames past the countdown it will take the coordinates and contour area average of the person when looking
# at the center and store it in an array to later be used.
# TODO: this module can be improved without the calibration module still, by using the area info that I haven't tried to use yet, it might be helpul to use it to detect y axis movements
# more accurately, as well as setting the conditions of the frame_count in the functions to input parameters instead of being hard coded like I did.

class Classifier():

	def __init__(self):
		self.centerAverage = [0, 0, 0]
		self.gazeAverage = [0, 0, 0]
		self.eyeCenterXAccumulator = 0
		self.eyeCenterYAccumulator = 0
		self.eyeCntAreaAccumulator = 0
		self.direction = ''

	def resetAccumulators(self):
		self.eyeCenterXAccumulator = 0
		self.eyeCenterYAccumulator = 0
		self.eyeCntAreaAccumulator = 0

	def increaseAccumulators(self, eye_center, eye_cnt):
		self.eyeCenterXAccumulator += eye_center[0]
		self.eyeCenterYAccumulator += eye_center[1]
		self.eyeCntAreaAccumulator += eye_cnt

	def findCenterAverage(self, frame_count):
		if 100 < frame_count < 180:
			self.centerAverage[0] = self.eyeCenterXAccumulator
			self.centerAverage[1] = self.eyeCenterYAccumulator
			self.centerAverage[2] = self.eyeCntAreaAccumulator

		elif frame_count == 180:
			self.centerAverage[0] = int(self.centerAverage[0] / 80)
			self.centerAverage[1] = int(self.centerAverage[1] / 80)
			self.centerAverage[2] = int(self.centerAverage[2] / 80)
			self.resetAccumulators()

	def findGazeAverage(self):
		self.gazeAverage[0] = int(self.eyeCenterXAccumulator / 5)
		self.gazeAverage[1] = int(self.eyeCenterYAccumulator / 5)
		self.gazeAverage[2] = int(self.eyeCntAreaAccumulator / 5)
		self.resetAccumulators()

	def classify(self, frame_count):
		if frame_count > 180 and frame_count % 5 == 0: 
			self.findGazeAverage()
			xDiff =  abs(self.gazeAverage[0] - self.centerAverage[0])
			yDiff =  abs(self.gazeAverage[1] - self.centerAverage[1])
			if xDiff > yDiff:
				if self.gazeAverage[0] < self.centerAverage[0]:
					self.direction = 'left'
				elif self.gazeAverage[0] > self.centerAverage[0]:
					self.direction = 'right'
			elif xDiff < yDiff:
				if self.gazeAverage[1] < self.centerAverage[1]:
					self.direction = 'up'
				elif self.gazeAverage[1] > self.centerAverage[1]:
					self.direction = 'down'
			else:
				self.direction = 'center'

if __name__ == "__main__":
    pass