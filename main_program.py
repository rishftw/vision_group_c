## COMP9517 Computer Vision Project 20T2
# main_program: Main program for the project
#
# Group C:
# Connor Baginski (z5207788)
# Bhumika Singhal (z5234799)
# Rishav Guha (z5294757)
# Amel Johny (z5294308)

from cell_tracking.helpers import plot_rectangles, find_centers, print_tracks, put_text, plot_rectangles_normal
from cell_tracking.pathTracking import PathTracker
from processing import *
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

def track_frame(image, tracker, labels, bounding_box_list, centers_list, frames):
    """
    Tracks the frame and records its bounding boxes and centers
    """
    centers, bounding_boxes, circular, is_circular = find_centers(labels, image)
    number_of_cells_in_frame = len(centers)
    put_text(image, 10, 100, "No of cells : {}".format(number_of_cells_in_frame))
    tracker.update(image, centers, circular, is_circular, bounding_boxes)
    bounding_box_list.append(bounding_boxes)
    plot_rectangles_normal(image, bounding_boxes)
    centers_list.append(centers)
    print_tracks(image, tracker)
    frames.append(image)

def show_frames(frames, tracker, bounding_box_list, centers_list):
    """
    Show each frame to the user
    """
    image_counter = 0
    for image in frames:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        plot_rectangles(image, bounding_box_list, tracker.mito_frames, image_counter)
        show_frame_control_display(tracker, image, image_counter, bounding_box_list, centers_list)
        image_counter += 1

def show_frame_control_display(tracker, image, frame, bounding_box_list, centers_list):
    """
    Display the frame and allow user pause/select cell
    """
    pause = False

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
                for i in range(len(bounding_box_list[frame])):
                    box = bounding_box_list[frame][i]
                    if x >= box[0] and box[2] >= x:
                        if y >= box[1] and box[3] >= y:
                            selectedBox = box
                            center = centers_list[frame][i]
                            break
                if selectedBox is not None:
                    # Display info
                    put_text(info, 10, info.shape[0], "Speed: {:.3f}".format(get_cell_speed_at_frame(tracker, frame, center)))
                    put_text(info, 10, info.shape[0]-20, "Distance Travelled: {:.3f}".format(get_cell_distance_at_frame(tracker, frame, center)))
                    put_text(info, 10, info.shape[0]-40, "Net Travelled: {:.3f}".format(get_cell_net_distance_at_frame(tracker, frame, center)))
                    put_text(info, 10, info.shape[0]-60, "Confinement Ratio: {:.3f}".format(get_cell_confinement_ratio_at_frame(tracker, frame, center)))
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

def detect_Fluo(image):
    # Threshold at value of 129
    thresh = cv2.threshold(image, 129, 255, cv2.THRESH_BINARY)[1]
    distance = ndi.distance_transform_edt(thresh)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((10, 10)),
                            labels=thresh)
    markers, _ = ndi.label(local_maxi)
    ws_labels = watershed(-distance, markers, mask=thresh)
    return ws_labels

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

def track_DIC():
    # Load CNN models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Network()
    # TODO: CHANGE TO BEST MODEL
    net.load_state_dict(torch.load("CNN_DIC.pth", map_location=device))
    
    # Initialise Tracker
    pathTracker = PathTracker(cost_threshold=10)
    bounding_box_list, centers_list = [], []
    frames = []
    # Load each frame
    for filename in fi_list('DIC-C2DH-HeLa/Sequence 3'):
        if not filename.endswith(".tif"):
            continue
        print(filename)
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        ws_labels = detect_DIC(image, net)
        track_frame(image, pathTracker, ws_labels, bounding_box_list, centers_list, frames)
    
    # Show all the frames
    show_frames(frames, pathTracker, bounding_box_list, centers_list)

def track_Fluo():
    # Initialise Tracker
    pathTracker = PathTracker(cost_threshold=10)

    bounding_box_list, centers_list = [], []
    frames = []
    # Load each frame
    for filename in fi_list('Fluo-N2DL-HeLa/02'):
        if not filename.endswith(".tif"):
            continue
        print(filename)
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        ws_labels = detect_Fluo(image)
        track_frame(image, pathTracker, ws_labels, bounding_box_list, centers_list, frames)
    
    # Show all the frames
    show_frames(frames, pathTracker, bounding_box_list, centers_list)

def track_PhC():
    # Load CNN models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Network()
    # TODO: CHANGE TO BEST MODEL
    net.load_state_dict(torch.load("CNN_PhC.pth", map_location=device))
    
    # Initialise Tracker
    pathTracker = PathTracker(cost_threshold=10)
    bounding_box_list, centers_list = [], []
    frames = []
    # Load each frame
    for filename in fi_list('PhC-C2DL-PSC/Sequence 3'):
        if not filename.endswith(".tif"):
            continue
        print(filename)
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        ws_labels = detect_PhC(image, net)
        track_frame(image, pathTracker, ws_labels, bounding_box_list, centers_list, frames)
    
    # Show all the frames
    show_frames(frames, pathTracker, bounding_box_list, centers_list)


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
