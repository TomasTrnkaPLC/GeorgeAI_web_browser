# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
import numpy as np



#load network
# construct output path
output_path = os.path.dirname(__file__)
default_pre_trained_model  = os.path.join(output_path, 'data\G1.cfg')
default_deploy_prototxt_file  = os.path.join(output_path, 'data\G1.weights')
default_deploy_names  = os.path.join(output_path, 'data\G1.names')
net = cv2.dnn.readNetFromDarknet( default_pre_trained_model, default_deploy_prototxt_file)
classes = []
with open(default_deploy_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
#Another data for detection
confidence_level = 0.15
detect_resolution = 0.50
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()
main_mode = "Object"
# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

@app.route('/object')
def object():
    
    print("The email address is ")
    return render_template("index.html")

@app.route('/face')
def face():
    
    print("object ")
    return render_template("index.html")

def george_core(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock


	# loop over frames from the video stream
	
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=800)
		if main_mode == "Object" :
			(h, w) = frame.shape[:2]
			# Detecting objects
			blob = cv2.dnn.blobFromImage(frame,  1 / 255.0, (320, 320), True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)
			boxes = []
			confidences = []
			classIDs = []
				# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
						# extract the class ID and confidence (i.e., probability)
						# of the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]
					if confidence > confidence_level:
						box = detection[0:4] * np.array([w, h, w, h])
						(centerX, centerY, width, height) = box.astype("int")
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)                  
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_level, detect_resolution)
				# ensure at least one detection exists
			if len(idxs) > 0:
					# loop over the indexes we are keeping
				for i in idxs.flatten():
						# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					color = [int(c) for c in colors[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.0%}".format(classes[classIDs[i]], confidences[i])
					cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)			 
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# grab the current timestamp and draw it on the frame
		
		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/', methods=["GET","POST"])
def home():    
    if request.method == 'POST':

        post_id = request.form.get('delete')

        if post_id is not None:
            print("Test")
            return redirect("/")
    		

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=george_core, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host='127.0.0.1', port=8080, debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()