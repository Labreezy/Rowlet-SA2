import sys
import os.path as osp
import cv2
import numpy as np
import imutils
from tqdm import tqdm


min_pause_sz_ratio = 60/720
max_pause_sz_ratio = 400/720
pause_aspect_ratio = 370/360
gc_pause_aspect_ratio = 375/275
BLUE = 0
RED = 2
HSV_MIN = np.array([95,200,200])
HSV_MAX = np.array([105,255,255])

def get_contour_box_size(c):
    _, _, w, h = cv2.boundingRect(c)
    return w * h
def sort_contours_by_size(contours):
    sorted_contours = sorted(contours, key=lambda c: get_contour_box_size(c), reverse=True)
    return sorted_contours

EPS = .05
def detect_box(frame,box_color=BLUE,gamecube=False):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_MIN,HSV_MAX)
    if np.sum(mask) == 0:
        return (False, frame)

    # find contours
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    n_contours = len(cnts)
    if n_contours == 0:
        return (False, frame)
    found_contour = False
    for i,c in enumerate(cnts):

        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if w < 120 or h < 120:
            continue

        aspect_ratio = w / h
        area_ratio = area/(w*h)
        if 0.07 < area_ratio < 0.125 and aspect_ratio > 0.95: #heuristically obtained constants
            print(f"Contour area ratio: {area_ratio} ({w}x{h})")
            debug_im = cv2.drawContours(frame, [c], -1, (0,255,0), 3)
            return (True, debug_im)
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
vid_fps = cap.get(cv2.CAP_PROP_FPS)
print(vid_width,vid_height)

min_pause_sz_ratio = 160/720
max_pause_sz_ratio = 380/720
last_paused_status = False
out=cv2.VideoWriter(out_path, -1, cap.get(cv2.CAP_PROP_FPS), (int(vid_width),int(vid_height)))
cap.set(cv2.CAP_PROP_POS_MSEC,39.0*1000)
frames_paused = 0
run_length_ms = (35*60+34.11)*1000
run_length_frames = int(run_length_ms/1000 * vid_fps)

pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        curr_pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        paused, newframe = detect_box(frame,BLUE,gamecube=True)

        if paused:
            frames_paused += 1
            newframe_txt = cv2.putText(newframe, f"Frames Paused: {frames_paused}", (0,0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (1.0,1.0,1.0),2,cv2.LINE_AA, False)
            for _ in range(int(vid_fps*.75)):
                out.write(newframe_txt)
            if not last_paused_status:
                last_paused_status = True

                print(f"PAUSE STARTED AT {int(curr_pos_msec*1000)}s")
                cv2.imshow("pause", newframe)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()

        else:
            #out.write(frame)
            if last_paused_status:
                last_paused_status = False
                print(f"PAUSE ENDED AT {pos_frame/vid_fps}s")

    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        cv2.waitKey(1000)
    pbar.update(1)
    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= run_length_frames + vid_fps*40:
        break
    if cap.get(cv2.CAP_PROP_POS_MSEC) >= (640*1000):
        break
pbar.close()
out.release()
cap.release()
print(f"Output to {out_path}")
print(f"Total frames paused: {frames_paused} @ {vid_fps} FPS")
input("Press Enter to Exit")

