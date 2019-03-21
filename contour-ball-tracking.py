#import packages
from collections import deque
from imutils.video import VideoStream
import numpy as np 
import argparse
import cv2
import imutils
import time

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-v", "--video", help="path to video file [optional]")
argument_parser.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(argument_parser.parse_args())

orange_color_lower = (0, 56, 169)
orange_color_upper = (77, 255, 255)

points = deque(maxlen=args['buffer'])

if not args.get("video", False):
    vs = VideoStream(src=0).start()

else:
    vs = cv2.VideoStream(args["video"])

time.sleep(2.0)

while True:
    frame = vs.read()

    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break
    
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, orange_color_lower, orange_color_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:

            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    points.appendleft(center)

    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        thickness = int()

        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key  == ord("q"):
        break

if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv2.destoryAllWindows()