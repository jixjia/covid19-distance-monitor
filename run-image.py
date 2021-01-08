# USAGE
# python run.py -i pedestrians.mp4
# python run.py -i pedestrians.mp4 -o output.avi

# import the necessary packages
from utils import config
from utils.detection import detect_people
from utils.overlay_bbox import transparent_box
from scipy.spatial import distance as dist
import cv2
import numpy as np
import argparse
import imutils
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
    help="path to (optional) input video file")
args = vars(ap.parse_args())

# load the COCO class labels (YOLO model)
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading model from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream (MIPI-CSI)...")
frame = cv2.imread(args["input"])

# resize the frame and then detect people (and only people) in it
frame = imutils.resize(frame, width=700)
h, w = frame.shape[:2]
results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

# initialize the set of indexes that violate the minimum social
# distance
violate = set()

# ensure there are *at least* two people detections (required in
# order to compute our pairwise distance maps)
if len(results) >= 2:
    # extract all centroids from the results and compute the
    # Euclidean distances between all pairs of the centroids
    centroids = np.array([r[2] for r in results])
    D = dist.cdist(centroids, centroids, metric="euclidean")

    # loop over the upper triangular of the distance matrix
    for i in range(0, D.shape[0]):
        for j in range(i + 1, D.shape[1]):
            # check to see if the distance between any two
            # centroid pairs is less than the configured number
            # of pixels
            if D[i, j] < config.MIN_DISTANCE:
                # update our violation set with the indexes of
                # the centroid pairs
                violate.add(i)
                violate.add(j)

# loop over the results
for (i, (prob, bbox, centroid)) in enumerate(results):
    # extract the bounding box and centroid coordinates, then
    # initialize the color of the annotation
    (startX, startY, endX, endY) = bbox
    (cX, cY) = centroid
    color = (0, 255, 0)

    # if the index pair exists within the violation set, then
    # update the color
    if i in violate:
        color = (0, 0, 255)

    # draw (1) a bounding box around the person and (2) the
    # centroid coordinates of the person,
    # cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    cv2.circle(frame, (cX, cY), 5, color, 1)
    frame = transparent_box(frame, bbox, color)

# draw the total number of social distancing violations on the
# output frame
frame = transparent_box(frame, (0, h-50, 150, h-10), (10,10,10))
violationPerc = len(violate) / float(len(results)) * 100
text = f"Violations: {len(violate)}".format(len(violate))
cv2.putText(frame, text, (10, h-25),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# show the output frame
cv2.imshow("Result", frame)
key = cv2.waitKey(0) & 0xFF

# if the `q` key was pressed, break from the loop
if key == ord("q"):
    cv2.destroyAllWindows()