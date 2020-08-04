
import os
from skimage.morphology import watershed, disk
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import meijering
import imutils
import matplotlib.pyplot as plt

##HELPER  FUNCTIONS

#extract centers of each labels 
def get_centers_and_boxes(ws_labels, image):
    centers = []
    boxes  = []
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
        # draw a rectangle enclosing the object
        try:
            x,y,w,h = cv2.boundingRect(c)
            center = np.array([[int(x + w / 2.0)], [int(y + h / 2.0)]])
            boxes.append([x,y,x+w,y+h])
            centers.append(center)
            
        except ZeroDivisionError:
            pass
    return centers, boxes

#plot rectangles around the labels 
def plot_rectangles(image, boundingBoxes):
    for i in range(len(boundingBoxes)):
        x1,y1,x2,y2 = boundingBoxes[i]
        cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
        
        
        
#write labels related to each label
#             cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
##Need to provide labels
# def put_lables(image, centers, boundingBoxes):
#     for i in range(len(boundingBoxes)):
#         x,y,_,_ = boundingBoxes[i]
#         cv2.putText(image, "#{}".format(label), (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    


def draw_path(image, tracker):
    for i in range(len(tracker.tracked_cells)):
        if (len(tracker.tracked_cells[i].positions) > 1):
            x = int(tracker.tracked_cells[i].positions[-1][0, 0])
            y = int(tracker.tracked_cells[i].positions[-1][0, 1])
            tl = (x-10, y-10)
            br = (x+10, y+10)
            cv2.rectangle(image, tl, br, (0, 255, 0), 1)

            for k in range(1, len(tracker.tracked_cells[i].positions) - 1):
                x = int(tracker.tracked_cells[i].positions[k][0, 0])
                y = int(tracker.tracked_cells[i].positions[k][0, 1])
                cv2.circle(image, (x, y), 1, (0,255,0), -1)
            cv2.circle(image, (x, y), 2, (0,255,0), -1)
    plt.imshow(image)
    plt.show()