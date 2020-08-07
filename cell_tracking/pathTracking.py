from collections import deque
import numpy as np
from cell_tracking.kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

class TrackedCell():

    def __init__(self, center, cell_id):
        self.cell_id = cell_id  # identification of each track object
        self.KF = KalmanFilter(center)  # KF instance to track this object
        self.prediction = np.asarray(center).reshape(
            1, 2)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.positions = []  # cell positions
        self.added_in_frame  = None
        self.circularity = []
        self.is_in_mitosis = []
        self.boundingBoxes = []

    def get_confinement_ratio(self):
        result = None

        try:
            result = self.get_distance_at_frame() / self.get_net_distance_at_frame()
        except:
            result = 0

        return result

    def get_net_distance_at_frame(self):
        return np.sqrt( np.sum( ( self.positions[0] - self.positions[-1] ) ** 2 ) )

    def get_distance_at_frame(self):
        return sum( [self.get_dist_at_frame(frame_num) for frame_num in range(1, len(self.positions))] )

    def get_dist_at_frame(self, idx):
        if len(self.positions) == 1:
            return 0
        elif idx < 0:
            return 0
        else:
            return np.sqrt( np.sum( ( self.positions[idx] - self.positions[idx-1] ) ** 2 ) )

    def get_speed_at_frame(self):

        if len(self.positions) == 1:
            return 0
        else:
            return np.sqrt( np.sum( ( self.positions[-1] - self.positions[-2] ) ** 2 ) )

    def updatePrediction(self, center):
        self.prediction = np.array(self.KF.predict()).reshape(1, 2)
        self.KF.correct(np.matrix(center).reshape(2, 1))

    def check_return_mitosis(self, offset):
        if(len(self.circularity)  < abs(offset)):
            return 0
        if self.circularity[offset] !=  True:
            return offset
        else:
            offset  = offset - 1
            return self.check_return_mitosis(offset)

    def get_mitosis_boxes(self,offset):
        boxes = [[]]
        # print(self.boundingBoxes)
        for i in range(len(self.positions) - offset-1,len(self.positions)):
            self.is_in_mitosis[i] = True
            # print(f'assigning bb: {self.boundingBoxes[i]}')
            boxes.append(self.boundingBoxes[i])
        return boxes

class PathTracker():

    def __init__(self, cost_threshold):
        
        self.tracker_id = 0
        self.frame_id = 0
        self.cost_thresh_allowed = cost_threshold
        self.max_frame_skips_allowed = 1
        self.max_trace_length_allowed = 1000
        # list of all tracked cell initialized
        self.tracked_cells = []
        # list of all tracked cell ID initialized
        self.tracked_cells_ids = []
        # list of all tracked cell initialized in frame
        self.tracks_cell_in_frame = []
        # list of all tracked cell initialized out of frame or disappeared
        self.tracks_blobs_in_frame = []
        # list of all tracked cell initialized out of frame or disappeared
        self.tracks_cell_disappeared_frame = []
        # list of all tracked cell initialized relation to parent
        self.tracks_cell_in_frame_parents = []
        #listof possible mitosis elements
        self.possible_motosis = []
        #Motisis cell frames
        self.mito_frames = []

    def get_cell_from_center(self, frame_id, center):
        N = len(self.tracks_cell_in_frame[frame_id])
        M = 1
        cost = [-1]*N
        for i in range(N):
            diff = self.tracks_cell_in_frame[frame_id][i].positions[frame_id - self.tracks_cell_in_frame[frame_id][i].added_in_frame] - center
            distance = np.sqrt(diff[0]*diff[0]  + diff[1]*diff[1])
            cost[i] = distance * (0.5)
        min_index = cost.index(min(cost))
        return self.tracks_cell_in_frame[frame_id][min_index]


    def add_cell(self, center, frame_id, circularity, boundingBox):
        # initialise new cell object
        new_cell = TrackedCell(center, self.tracker_id)
        # add cell ID to list of all tracked cell ID initialized
        self.tracked_cells_ids.append(self.tracker_id)
        # add given center to initial position
        new_cell.positions.append(center)
        # add cell to list of all tracked cells initialized
        self.tracked_cells.append(new_cell)
        # increment tracker ID
        self.tracker_id += 1
        # Add added in frame id
        new_cell.added_in_frame = self.frame_id
        # Add circularity status
        new_cell.circularity.append(circularity)
        # Add mitosis status
        new_cell.is_in_mitosis.append(False)
        # Add bounding box
        new_cell.boundingBoxes.append(boundingBox)
        return new_cell

    def add_to_frame(self, center):
        # add the frame to list of tracked cells per frame
        np.append(self.tracks_cell_in_frame[self.frame_id], center)

    def update(self, image, centers, circular, is_circular, boundingBoxes):
        # If centers are detected in frame process the centers
        cell_centers_in_frame = np.zeros((len(centers), 2), dtype="int")
        for blob in circular:    
            self.tracks_blobs_in_frame.append(blob)
        self.tracks_cell_disappeared_frame.append([])
        self.mito_frames.append([])
        print(len(self.mito_frames))
        if (len(cell_centers_in_frame) > 0):
            # Store the centers obtained in the frame to cell_centers_in_frame
            for (i, center) in enumerate(centers):
                cell_centers_in_frame[i] = center
            # If it  is the initial frame add all the centers as new cells
            if len(self.tracked_cells) == 0:
                for i in range(len(cell_centers_in_frame)):
                    # Add the cell and returns cell ID
                    self.add_cell(cell_centers_in_frame[i], self.frame_id, is_circular[i],boundingBoxes[i])
                # Add the centers list to the tracked centers list of the frame.

            # Calculate cost using sum of square distance between
            # predicted vs detected centroids
            # TODO: TO BE REFERENCED
            N = len(self.tracked_cells)
            # RETURN THE NUMBER OF CELL  ALREADY IN TRACKED LIST
            M = len(cell_centers_in_frame)
            # RETURN THE NUMBER OF CELL  IDENTIFIED IN NEW FRAME

            # Cost matrix giving distance cost with new centers found 
            cost = np.zeros(shape=(N, M))   
            least_finder = np.zeros(shape=(N, M))
            for i in range(N):
                for j in range(M):
                    try:
                        diff = self.tracked_cells[i].prediction - \
                            cell_centers_in_frame[j]
                        distance = np.sqrt(
                            diff[0][0]*diff[0][0] + diff[0][1]*diff[0][1])
                        cost[i][j] = distance
                        least_finder[i][j] = distance
                    except:
                        print("error while calculating distance")
                        pass

            # Let's average the squared ERROR
            cost = (0.5) * cost
            # Using Hungarian Algorithm assign the correct detected measurements
            # to predicted cell positions
            row_ind, col_ind = linear_sum_assignment(cost)

            assignment = [-1 for i in range(N)]
            for i in range(len(row_ind)):
                assignment[row_ind[i]] = col_ind[i]

            # Identify tracker with no assignment, if any
            un_assigned_tracks = []
            for i in range(len(assignment)):
                if (assignment[i] != -1):
                    # check for cost distance threshold.
                    # If cost is very high then un_assign (delete) the tracker.track
                    if (cost[i][assignment[i]] > self.cost_thresh_allowed):
                        assignment[i] = -1
                        un_assigned_tracks.append(i)
                        self.tracked_cells[i].skipped_frames += 1
                    else:
                        pass
                else:
                    self.tracked_cells[i].skipped_frames += 1
            
            # If tracker are not detected for long time, remove them
            del_tracks = []
            for i in range(len(self.tracked_cells)):
                if (self.tracked_cells[i].skipped_frames > 0):
                    del_tracks.append(i)

            # check for mitosis
            possible_mitosis_in_frame = []
            if len(del_tracks) > 0:  # only when skipped frame exceeds max
                offset = 0
                for id in del_tracks:
                    if id < N:
                        self.tracks_cell_disappeared_frame[self.frame_id].append(self.tracked_cells[id-offset].cell_id)
                        cost = [-1]*len(self.tracks_blobs_in_frame)
                        for index in range(len(self.tracks_blobs_in_frame)):
                             
                            diff  = self.tracks_blobs_in_frame[index] - self.tracked_cells[id-offset].positions[-1]
                            distance = np.sqrt(
                                diff[0]*diff[0] + diff[1]*diff[1])
                            if distance < 8:
                                possible_mitosis_in_frame.append(self.tracked_cells[id-offset].cell_id)
                                # print(self.tracked_cells[id-offset].cell_id,possible_mitosis_in_frame, abs(self.tracked_cells[id-offset].check_return_mitosis(-1)))
                                result = abs(self.tracked_cells[id-offset].check_return_mitosis(-1))
                                if result>0:
                                    mito_boxes = self.tracked_cells[id-offset].get_mitosis_boxes(result)
                                    for length in range(len(mito_boxes)):
                                        box = mito_boxes.pop()
                                        if box not in self.mito_frames[self.frame_id-length]:
                                            self.mito_frames[self.frame_id-length].append(box)

                        del self.tracked_cells_ids[id-offset]
                        del self.tracked_cells[id-offset]
                        del assignment[id-offset]
                        offset+=1
                    else:
                        print("ERROR: id is greater than length of tracker.tracks")
            self.possible_motosis.append(possible_mitosis_in_frame)
            # Now look for un_assigned detects
            un_assigned_detects = []
            for i in range(len(cell_centers_in_frame)):
                if i not in assignment:
                    un_assigned_detects.append(i)

            # Start new tracker for the cells
            if(len(un_assigned_detects) != 0):
                for i in range(len(un_assigned_detects)):
                    added_cell = self.add_cell(
                        cell_centers_in_frame[un_assigned_detects[i]], self.frame_id,is_circular[un_assigned_detects[i]],boundingBoxes[un_assigned_detects[i]])

            for i in range(len(assignment)):
                self.tracked_cells[i].KF.predict()
                if assignment[i] > -1:
                    self.tracked_cells[i].circularity.append(is_circular[assignment[i]])
                    self.tracked_cells[i].is_in_mitosis.append(False)
                    
                    self.tracked_cells[i].skipped_frames = 0
                    self.tracked_cells[i].KF.state, self.tracked_cells[i].prediction = self.tracked_cells[i].KF.correct(
                        np.matrix(cell_centers_in_frame[assignment[i]]).reshape(2, 1), 1)
                    self.tracked_cells[i].boundingBoxes.append(boundingBoxes[assignment[i]])
                else:
                    self.tracked_cells[i].circularity.append(False)
                    self.tracked_cells[i].is_in_mitosis.append(False)
                    self.tracked_cells[i].KF.state, self.tracked_cells[i].prediction = self.tracked_cells[i].KF.correct(
                        np.matrix([[0], [1]]).reshape(2, 1), 0)

                self.tracked_cells[i].positions.append(
                    cell_centers_in_frame[assignment[i]])

        self.tracks_cell_in_frame.append(deepcopy(self.tracked_cells))
        self.frame_id += 1
        return
