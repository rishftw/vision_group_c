
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

#File read
def fi_list(path):
    """
    Return a sorted list of filenames in a given path
    """
    return sorted([os.path.join(path, f) for f in os.listdir(path)])


#threshold
def custom_thresh(image):
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if image[r,c] > 0.05:
                image[r,c] = 255
            else:
                image[r,c] = 0

    return image


#Erode
def disk_erode(img, radius=24, iters=1):
    image = img.copy()
    kern_disk = disk(radius)
    eroded = cv2.erode(image, kern_disk, iterations=iters)
    
    return eroded


#Extract labels
def find_labels(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    thresh = cv2.threshold(image, 162, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((4,4),np.uint8)
    # Perform an erosion followed by dilation opening to remove noise
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    distance = ndi.distance_transform_edt(opening)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((10, 10)),
                            labels=thresh)
    markers, _ = ndi.label(local_maxi)
    ws_labels = watershed(-distance, markers, mask=thresh)
    return ws_labels,image


#extract centers of each labels 
def find_centers(ws_labels, image):
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
            center = (int(x + w / 2.0), int(y + h / 2.0))
            boxes.append([x,y,x+w,y+h])
            centers.append(center)
            
        except ZeroDivisionError:
            pass
    return centers, boxes



#plot rectangles around the labels 
def plot_rectangles(image, centers, boundingBoxes):
    for i in range(len(boundingBoxes)):
        x1,y1,x2,y2 = boundingBoxes[i]
        cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)\
        
        
        
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