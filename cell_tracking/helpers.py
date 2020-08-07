
import os
from skimage.morphology import watershed, disk
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import meijering
import imutils
import matplotlib.pyplot as plt
import math

## HELPER  FUNCTIONS

# Extract centers of each labels 
def find_centers(ws_labels, image):
    centers = []
    boxes  = []
    pi_4 = np.pi * 4
    circular = []
    is_circular = []
    for label in np.unique(ws_labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
    
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(image.shape, dtype="uint8")
        mask[ws_labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area <= 0:  # skip ellipses smaller then 10x10
            continue

        arclen = cv2.arcLength(c, True)
        circularity = (pi_4 * area) / (arclen * arclen)
        
        # draw a rectangle enclosing the object
        try:
            x,y,w,h = cv2.boundingRect(c)
            M = cv2.moments(c)
            if M["m00"] and M["m00"]:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # center = (int(x + w / 2.0), int(y + h / 2.0))
                centers.append(center)
                boxes.append([x,y,x+w,y+h])
                if(circularity > 0.80):
                    circular.append(np.asarray(center))
                    is_circular.append(True)
                else:
                    is_circular.append(False)

        except ZeroDivisionError:
            pass
    return centers, boxes, circular, is_circular

# plot rectangles around the labels 
def plot_rectangles(image, boundingBoxesList, mito_frames, image_index):
    counter  = 0
    for i in range(len(boundingBoxesList[image_index])):
        x1,y1,x2,y2 = boundingBoxesList[image_index][i]
      
    try:
    
        if boundingBoxesList[image_index][i] in mito_frames[image_index+1]:
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            counter += 1
        else:
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,0),2)
    except:
        pass
    put_text(image,10,50, f'Mitosis Count: {counter}')

# plot rectangles around the labels 
def plot_rectangles_normal(image, boundingBoxes):
    for i in range(len(boundingBoxes)):
        x1,y1,x2,y2 = boundingBoxes[i]
        cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
        
# puts text on image
def put_text(image, x,y,text):
    cv2.putText(image,text, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
# Draws the path from the given tracking object 
def print_tracks(plot_image,tracker):
    for i in range(len(tracker.tracked_cells )):
        if (len(tracker.tracked_cells [i].positions) > 1):                
            for k in range(1, len(tracker.tracked_cells [i].positions) - 1):
                x = int(tracker.tracked_cells [i].positions[k][0])
                y = int(tracker.tracked_cells [i].positions[k][1])
                x2 = int(tracker.tracked_cells [i].positions[k+1][0])
                y2 = int(tracker.tracked_cells [i].positions[k+1][1])
                
                cv2.line(plot_image, (x, y), (x2,y2), (0,255,0), 2)