## COMP9517 Computer Vision Project 20T2
# main_program: Main program for the project
#
# Group C:
# Connor Baginski (z5207788)
# Bhumika Singhal (z5234799)
# Rishav Guha (z5294757)
# Amel Johny (z5294308)

from cell_tracking.helpers import plot_rectangles, get_centers_and_boxes
from cell_tracking.pathTracking import PathTracker
from processing import get_ws_from_markers, equalize_clahe
from network import Network

import numpy as np
import cv2
import os
import imutils

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.morphology import watershed, disk
from skimage.feature import peak_local_max

from scipy import ndimage as ndi

track_colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
font = cv2.FONT_HERSHEY_SIMPLEX
selectPos = None
selectFlag = False

def fi_list(path):
    """
    Return a sorted list of filenames in a given path
    """
    return sorted([os.path.join(path, f) for f in os.listdir(path)])

def onMouse(event, x, y, flags, param):
    global selectPos, selectFlag
    if event == cv2.EVENT_LBUTTONDOWN:
        selectPos = (x, y)
        selectFlag = True

def draw_tracks(tracker, image, labels, frame):
    pause = False
    centers, bounding_boxes = get_centers_and_boxes(labels, image)
    # draw bounding boxes for the detected cells
    plot_rectangles(image, bounding_boxes)
    tracker.Update(centers)
    for i in range(len(tracker.tracks)):
        if len(tracker.tracks[i].trace) > 1:
            for j in range(len(tracker.tracks[i].trace)-1):
                # Draw trace line
                x1 = tracker.tracks[i].trace[j][0][0]
                y1 = tracker.tracks[i].trace[j][1][0]
                x2 = tracker.tracks[i].trace[j+1][0][0]
                y2 = tracker.tracks[i].trace[j+1][1][0]
                clr = tracker.tracks[i].track_id % 9
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                         track_colours[clr], 3)
    cv2.namedWindow('Path Tracker')
    cv2.setMouseCallback('Path Tracker', onMouse)
    cv2.imshow('Path Tracker', image)

    global selectPos, selectFlag
    key = cv2.waitKey(250) & 0xff
    if key == 27:  # 'Esc' key has been pressed, exit program.
        exit()
    if key == 32:  # 'Space' has been pressed. Pause/Resume
        pause = not pause
        print("Code is paused. Press 'space' to resume.")
        while (pause is True):
            if selectFlag is True: # User has selected a position on the image
                info = image.copy()
                # Find selected cell
                selectedBox = None
                x, y = selectPos
                for i in range(len(bounding_boxes)):
                    box = bounding_boxes[i]
                    if x >= box[0] and box[2] >= x:
                        if y >= box[1] and box[3] >= y:
                            selectedBox = box
                            center = centers[i]
                            break
                if selectedBox is not None:
                    # TODO Use "frame" param to get info for task 3
                    # Display info
                    global font
                    cv2.putText(info, "Centre: ({}, {})".format(center[0][0], center[1][0]), (10, info.shape[0]-10),
                                font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(info, "Centre: ({}, {})".format(center[0][0], center[1][0]), (10, info.shape[0]-30),
                                font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(info, "Centre: ({}, {})".format(center[0][0], center[1][0]), (10, info.shape[0]-50),
                                font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(info, "Centre: ({}, {})".format(center[0][0], center[1][0]), (10, info.shape[0]-70),
                                font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Path Tracker', info)
                selectFlag = False
            key = cv2.waitKey(50) & 0xff
            if key == 32:
                pause = False
                print("Resuming...")
                break
            if key == 27:
                exit()

def detect_DIC(image, net):
    # Preprocessing
    x = equalize_clahe(image)
    x = torch.tensor(np.array([x.astype(np.float32)]))
    # Add a "Batch" dimension
    x = x.unsqueeze(0)

    with torch.no_grad():
        net.eval()
        # Generate cell mask and markers from image
        output = net(x)
        markers = (output[0,0] > 0.5).int()
        cell_mask = (output[0,1] > 0.5).int()
        
        # Postprocessing
        ws_labels = get_ws_from_markers(markers.numpy(), cell_mask.numpy(), 12)
        
    return ws_labels

def track_DIC():
    # Load CNN models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Network()
    # TODO: CHANGE TO BEST MODEL
    net.load_state_dict(torch.load("CNN_min_loss_dic.pth", map_location=device))
    
    # Initialise Tracker
    tracker = PathTracker(20, 30, 15, 100)
    frame = 0
    for filename in fi_list('DIC-C2DH-HeLa/Sequence 3'):
        if not filename.endswith(".tif"):
            continue
        print(filename)
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)        
        ws_labels = detect_DIC(image, net)
    
        draw_tracks(tracker, image, ws_labels, frame)
        frame += 1

def detect_Fluo(image):
    # Threshold at value of 129
    thresh = cv2.threshold(image, 129, 255, cv2.THRESH_BINARY)[1]
    distance = ndi.distance_transform_edt(thresh)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((10, 10)),
                            labels=thresh)
    markers, _ = ndi.label(local_maxi)
    ws_labels = watershed(-distance, markers, mask=thresh)
    return ws_labels

def track_Fluo():
    # Initialise Tracker
    tracker = PathTracker(20, 30, 15, 100)
    frame = 0
    for filename in fi_list('Fluo-N2DL-HeLa/01'):
        if not filename.endswith(".tif"):
            continue
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)        
        ws_labels = detect_Fluo(image)

        draw_tracks(tracker, image, ws_labels, frame)
        frame += 1

def detect_PhC(image, net):
    # Preprocessing
    image = equalize_clahe(image)
    x = torch.tensor(np.array([image.astype(np.float32)]))
    # Add a "Batch" dimension
    x = x.unsqueeze(0)

    with torch.no_grad():
        net.eval()
        # Generate cell mask and markers from image           
        output = net(x)
        markers = (output[0,0] > 0.5).int()
        cell_mask = (output[0,1] > 0.5).int()
        
        # Postprocessing
        ws_labels = get_ws_from_markers(markers.numpy(), cell_mask.numpy(), 0)

    return ws_labels

def track_PhC():
    # Load CNN models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Network()
    # TODO: CHANGE TO BEST MODEL
    net.load_state_dict(torch.load("CNN_min_loss_phc.pth", map_location=device))
    
    # Initialise Tracker
    tracker = PathTracker(20, 30, 15, 100)
    frame = 0
    for filename in fi_list('PhC-C2DL-PSC/Sequence 3'):
        if not filename.endswith(".tif"):
            continue
        print(filename)
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        ws_labels = detect_PhC(image, net)
            
        draw_tracks(tracker, image, ws_labels, frame)
        frame += 1

def main():
    select = int(input("Choose a dataset.\n1) DIC-C2DH-HeLa\n2) Fluo-N2DL-HeLa\n3) PhC-C2DL-PSC\n> "))
    # select = 3 ##FOR TESTING

    if select == 1:
        track_DIC()
    elif select == 2:
        track_Fluo()
    elif select == 3:
        track_PhC()
    else:
        print("Invalid input.")

if __name__ == '__main__':
    main()
