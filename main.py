import sys
import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt


min_pause_sz_ratio = 60/720
max_pause_sz_ratio = 400/720
pause_aspect_ratio = 370/360
def get_contour_box_size(c):
    _, _, w, h = cv2.boundingRect(c)
    return w * h
def sort_contours_by_size(contours):
    sorted_contours = sorted(contours, key=lambda c: get_contour_box_size(c), reverse=True)
    return sorted_contours
def detect_box(frame):
    frame_blue = cv2.extractChannel(frame, 0)
    blurred = cv2.GaussianBlur(frame_blue, (9,9), 0)
    t, thresh = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY_INV)
    edged=cv2.Canny(thresh, 30, 200)

    # find contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    n_contours = len(contours)
    if n_contours == 0:
        return (False, frame)
    sorted_contours = sort_contours_by_size(contours)

    min_pause_size = frame.shape[0] * min_pause_sz_ratio
    max_pause_size = frame.shape[0] * max_pause_sz_ratio

    for i in range(len(sorted_contours)):
        c = sorted_contours[i]
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10:
            return (False, frame)

        aspect_ratio = w / h
        ar_diff = abs(aspect_ratio - pause_aspect_ratio)

        if ar_diff < 0.05 and w > min_pause_size and w < max_pause_size:
            return (True, cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), -1))
    return (False, frame)

if len(sys.argv) > 1:
    in_path = sys.argv[1]
    out_path = osp.splitext(in_path)[0] + "_out.mp4"
else:
    print("Drag and drop a video file here, dummy")
    input("Press any key to exit")
    sys.exit(0)
cap = cv2.VideoCapture(in_path)
while not cap.isOpened():
    cap = cv2.VideoCapture(in_path)
    cv2.waitKey(1000)
    print("waiting for header")

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(vid_width,vid_height)

min_pause_sz_ratio = 160/720
max_pause_sz_ratio = 380/720

out=cv2.VideoWriter(out_path, -1, cap.get(cv2.CAP_PROP_FPS), (int(vid_width),int(vid_height)))
while True:
    ret, frame = cap.read()
    if ret:
        pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        paused, newframe = detect_box(frame)

        if paused:
            print(f"PAUSE AT FRAME {pos_frame}, skipping")
        else:
            out.write(frame)


    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        cv2.waitKey(1000)
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break
out.release()
cap.release()
print(f"Output to {out_path}")
input("Press any key to exit.")

