# train_networks: Training CNN to be used by the main program
# The training currently uses the cached ima

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import imutils
from processing import *

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader

from skimage.segmentation import watershed
from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage.filters import meijering

from scipy import ndimage as ndi

BATCH_SIZE = 2
IMAGES_LIMIT = 5000

DROP_IN = 1000

"""
def plot_two_images(imgL, imgR, titleL, titleR):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(imgL, cmap='gray')
    plt.title(titleL)
    f.add_subplot(1,2, 2)
    plt.imshow(imgR, cmap='gray')
    plt.title(titleR)
    plt.show(block=True)
"""

# To load from drive
# def load_data(dataset):
#     data = []
#     paths = [os.path.join(dataset, '01'), os.path.join(dataset, '02')]
#     for path in paths:
#         mask_path = path + '_ST'
#         mask_path = os.path.join(mask_path, 'SEG')
#         for f in os.listdir(mask_path):
#             if not f.endswith(".tif"):
#                 continue
#             image = cv2.imread(os.path.join(path, f.replace('man_seg', 't')), cv2.IMREAD_GRAYSCALE)
#             image = equalize_clahe(image).astype(np.float32)
#             mask = cv2.imread(os.path.join(mask_path, f), cv2.IMREAD_UNCHANGED)
#             print("   Loaded " + os.path.join(mask_path, f) + ", " + os.path.join(path, f.replace('man_seg', 't')))
            
#             # Generate the Cell Mask and Markers from the Mask
#             cell_mask = (mask > 0).astype(np.uint8)
#             markers = (get_markers(mask) > 0).astype(np.uint8)
#             weight_map = get_weight_map(markers)
            
#             # Pack the data for the DataLoader
#             target = (cell_mask, markers, weight_map)
#             data.append((np.array([image]), target))

#     train_size = int(0.8 * len(data))
#     test_size = len(data) - train_size
#     train_data, test_data = random_split(data, [train_size, test_size])
#     trainLoader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
#     testLoader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
#     return trainLoader, testLoader

# Cache System (for large number of images)
def load_data(dataset):
    data = []
    path = os.path.join(dataset, "originals")
    clahe_path = path.replace("originals", "clahes")
    mask_path = path.replace("originals", "masks")
    markers_path = path.replace("originals", "markers")
    wm_path = path.replace("originals", "weight_maps")

    count = 0
    hid_cnt = 0
    for f in os.listdir(path):
        # Limits dataset size for RAM compatibility
        if count >= IMAGES_LIMIT:
            break
            
        # Limits dataset size for RAM compatibility
        hid_cnt += 1
        if hid_cnt % DROP_IN == 0:
            continue
        
        if not f.endswith(".npy"):
            continue
    
        count += 1

        clahe = np.load(os.path.join(clahe_path, f))
        cell_mask = np.load(os.path.join(mask_path, f))
        markers = np.load(os.path.join(markers_path, f))
        weight_map = np.load(os.path.join(wm_path, f))
        print("   Loaded " + os.path.join(mask_path, f) + ", " + os.path.join(path, f.replace('mask', '')))

        # Pack the data for the DataLoader
        target = (cell_mask, markers, weight_map)
        data.append((np.array([clahe]), target))
    
#     # For RAM
#     keep_size = int(KEEP_RATIO * len(data))
#     data, _ = random_split(data, [keep_size, len(data) - keep_size])
    
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    trainLoader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    return trainLoader, testLoader

# Class for creating the CNN
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv11 = nn.Conv2d(768, 256, 3, padding=1)
        self.conv12 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv13 = nn.Conv2d(384, 128, 3, padding=1)
        self.conv14 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv15 = nn.Conv2d(192, 64, 3, padding=1)
        self.conv16 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv17 = nn.Conv2d(96, 32, 3, padding=1)
        self.conv18 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv_out = nn.Conv2d(32, 2, 1)
        
    def forward(self, x):
        """
        Forward pass through the network
        """
        # Convolutional Layer
        x = F.relu(self.conv1(x))
        contraction_32 = F.relu(self.conv2(x))
        
        # Max Pooling Layer
        x = F.max_pool2d(contraction_32, kernel_size=2)
        x = F.relu(self.conv3(x))
        contraction_64 = F.relu(self.conv4(x))
        
        x = F.max_pool2d(contraction_64, kernel_size=2)
        x = F.relu(self.conv5(x))
        contraction_128 = F.relu(self.conv6(x))
        
        x = F.max_pool2d(contraction_128, kernel_size=2)
        x = F.relu(self.conv7(x))
        contraction_256 = F.relu(self.conv8(x))
        
        x = F.max_pool2d(contraction_256, kernel_size=2)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        
        # Upscaling layer
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # Concatenation Layer
        x = torch.cat((contraction_256, x), dim=1)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((contraction_128, x), dim=1)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((contraction_64, x), dim=1)
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((contraction_32, x), dim=1)
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        
        # Output Convolutional Layer
        x = self.conv_out(x)
        # Sigmoid Activation for predicted value in (0, 1)
        output = torch.sigmoid(x)
        return output

def weighted_mean_sq_error(inputs, targets_m, targets_c, weights):

#     Weighted Mean Squared Error Loss takes in a weight map
#     and computes loss based on both outputs
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('.', end='')
    
    inputs = inputs.to(device)
    targets = [targets_m.to(device), targets_c.to(device)]
    weights = weights.to(device)
    loss = torch.zeros(inputs.shape[0])

    # Calculate loss for each sample in the batch
    for sample in range(inputs.shape[0]):
        sample_loss, total_weight = 0.0, 0.0
        
        pred_markers = inputs[sample][0]
        pred_cmask = inputs[sample][1]
        
        exp_markers = targets[0][sample]
        exp_cmask = targets[1][sample]
        
        numerator = weights[sample] * ( (pred_markers-exp_markers)**2 + (pred_cmask-exp_cmask)**2 )        
        numerator = torch.sum(numerator)

        denominator = torch.sum(weights[sample])

        sample_loss = 0.5 * (numerator/denominator)        
        loss[sample] = sample_loss

    return torch.mean(loss)

"""
def get_score(outputs, ground_truth):
    
    #Calculates Accuracy Score across the batch
    
    score = 0
    batch_size = outputs.shape[0]
    total = outputs.shape[1] * outputs.shape[2]
    for sample in range(batch_size):
        num_correct = torch.sum(outputs[sample] == ground_truth[sample]).item()
        score += float(num_correct) / total

    return score / batch_size
"""


def main():
    """
    Train 2 networks for predicting markers and the cell mask respectively
    Set trains on data from "Sequence 1 Masks" and "Sequence 2 Masks"
    and save the models
    """ 
    print("Loading Data...")
    # Change this to the training set to be loaded
    trainLoader, testLoader = load_data('DIC-3_cache')
    print("Finished.")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    
    # Net predicts the markers and cells
    net = Network().to(device)
    
    # Calculate loss with Weighted MSE
    criterion = weighted_mean_sq_error
    
    # Optimising using Adam algorithm
    optimiser = optim.Adam(net.parameters(), lr=0.001)
    
    min_loss = 9999999999
    file_count_c = 0
    
    # Iterate over a number of epochs on the data
    for epoch in range(100):
        for i, batch in enumerate(trainLoader, 0):
            x = batch[0].to(device)
            target = batch[1]
            cell_masks, markers = target[0].to(device), target[1].to(device) # Unpack target data
            weight_map = target[2].to(device)

            # Clear gradients from last step
            optimiser.zero_grad()
            
            # Predict the markers from the image
            output = net(x)
            loss = criterion(output, markers.float(), cell_masks.float(), weight_map)
            loss.backward()
            optimiser.step()

            # Adjust based on number of batches
            if i == 0 or (i + 1) % 960 == 0:
                print(f"Epoch: {epoch+1}, Batch: {i + 1}")
                print(f"Loss: {loss.item():.2f}")
                
                # plt.imshow(x[0][0].cpu(), cmap='gray')
                # plt.title("Input")
                # plt.show()

                # Get the predicted Cell Mask and Markers for one of the images
                pred_m = ( (output[0][0] > 0.5).int() ).cpu()
                pred_c = ( (output[0][1] > 0.5).int() ).cpu()
                
                # Compare predicted to true images
                # plot_two_images(pred_c.cpu(), cell_masks[0].cpu(), "Predicted Cell Mask", "True Cell Mask")
                # plot_two_images(pred_m.cpu(), markers[0].cpu(), "Predicted Markers", "True Markers")


        # Test on the evaluation set
        print("\n--- Evaluation ---")
        net.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, batch in enumerate(testLoader):
                x = batch[0].to(device)
                target = batch[1]
                cell_masks, markers = target[0].to(device), target[1].to(device) # Unpack target data
                weight_map = target[2].to(device)

                output = net(x)
                
                pred_m = (output[:,0] > 0.5).int()
                pred_c = (output[:,1] > 0.5).int()

                running_loss += weighted_mean_sq_error(output, markers.float(), cell_masks.float(), weight_map)

                # if i == 0:
                    # plt.imshow(x[0][0].cpu(), cmap='gray')
                    # plt.title("Input")
                    # plt.show()

                    # Compare predicted to true images
                    # plot_two_images(pred_c[0].cpu(), cell_masks[0].cpu(), "Predicted Cell Mask", "True Cell Mask")
                    # plot_two_images(pred_m[0].cpu(), markers[0].cpu(), "Predicted Markers", "True Markers")

            loss = running_loss / len(testLoader)
            
            if loss < min_loss:
                # Change this to desired model output
                torch.save(net.state_dict(), "./CNN_DIC.pth")
                print("Saved model at Epoch {}, Loss: {:.3f}".format(epoch, loss))
                min_loss = loss
            
            print(f"EPOCH {epoch+1} LOSS: {loss:.3f}\nMin Loss: {min_loss:.3f}\n\n")
        net.train()

if __name__ == '__main__':
    main()