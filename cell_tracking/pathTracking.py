from collections import deque
import numpy as np
from cell_tracking.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import cv2
import matplotlib.pyplot as plt


class TrackedCell():

    def __init__(self, center, cell_id):
        self.cell_id = cell_id  # identification of each track object
        self.KF = KalmanFilter(center)  # KF instance to track this object
        self.prediction = np.asarray(center).reshape(
            1, 2)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.positions = []  # cell positions

    def updatePrediction(self, center):
        """ Predict the state of the cell """
        self.prediction = np.array(self.KF.predict()).reshape(1, 2)
        self.KF.correct(np.matrix(center).reshape(2, 1))


class PathTracker():

    def __init__(self):
        """Class to keep variable used to Track cells for cell class
        """
        self.tracker_id = 0
        self.frame_id = 0
        self.cost_thresh_allowed = 150
        self.max_frame_skips_allowed = 10
        self.max_trace_length_allowed = 1000
        # list of all tracked cell initialized
        self.tracked_cells = []
        # list of all tracked cell ID initialized
        self.tracked_cells_ids = []
        # list of all tracked cell initialized in frame
        self.tracks_cell_in_frame = []
        # list of all tracked cell initialized out of frame or disappeared
        self.tracks_cell_disappeared_frame = []
        # list of all tracked cell initialized relation to parent
        self.tracks_cell_in_frame_parents = []

    def add_cell(self, center):
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
        return new_cell

    def add_to_frame(self, center):
        # add the frame to list of tracked cells per frame
        np.append(self.tracks_cell_in_frame[self.frame_id], center)

    def update(self, image, centers):
        # If centers are detected in frame process the centers
        cell_centers_in_frame = np.zeros((len(centers), 2), dtype="int")

        if (len(cell_centers_in_frame) > 0):
            # Store the centers obtained in the frame to cell_centers_in_frame
            for (i, center) in enumerate(centers):
                cell_centers_in_frame[i] = center
            # print(cell_centers_in_frame[0:10])
            # If it  is the initial frame add all the centers as new cells
            if len(self.tracked_cells) == 0:
                for i in range(len(cell_centers_in_frame)):
                    # Add the cell and returns cell ID
                    self.add_cell(cell_centers_in_frame[i])
                # Add the centers list to the tracked centers list of the frame.

            self.tracks_cell_in_frame.append(cell_centers_in_frame)

            # Calculate cost using sum of square distance between
            # predicted vs detected centroids
            # TODO: TO BE REFERENCED
            N = len(self.tracked_cells)
            M = len(cell_centers_in_frame)

            cost = np.zeros(shape=(N, M))   # Cost matrix
            for i in range(N):
                for j in range(M):
                    try:

                        diff = self.tracked_cells[i].prediction - \
                            cell_centers_in_frame[j]
                        distance = np.sqrt(
                            diff[0][0]*diff[0][0] + diff[0][1]*diff[0][1])
                        cost[i][j] = distance
                    except:
                        print("error while calculating distance")
                        pass

            # for i in range(len(self.tracked_cells)):
            #     for j in range(len(cell_centers_in_frame)):
            #         if int(cost[i][j]) == 0:
            #             print('cost:',i,'-> ',j , cost[i][j],'\n')
            # Let's average the squared ERROR

            cost = (0.5) * cost
            # Using Hungarian Algorithm assign the correct detected measurements
            # to predicted cell positions
            row_ind, col_ind = linear_sum_assignment(cost)

            assignment = [-1 for i in range(N)]

            for i in range(len(row_ind)):
                assignment[row_ind[i]] = col_ind[i]

            # Identify tracker.tracks with no assignment, if any
            un_assigned_tracks = []
            for i in range(len(assignment)):
                if (assignment[i] != -1):
                    #                     # check for cost distance threshold.
                    #                     # If cost is very high then un_assign (delete) the tracker.track
                    if (cost[i][assignment[i]] > self.cost_thresh_allowed):
                        assignment[i] = -1
                        un_assigned_tracks.append(i)
                    else:
                        pass

                else:
                    # un_assigned_tracks.append(i)
                    self.tracked_cells[i].skipped_frames += 1
            # If tracker.tracks are not detected for long time, remove them
            del_tracks = []
            for i in range(len(self.tracked_cells)):
                if (self.tracked_cells[i].skipped_frames > self.max_frame_skips_allowed):
                    del_tracks.append(i)
            # del_tracks = sorted(del_tracks, reverse=True)
            if len(del_tracks) > 0:  # only when skipped frame exceeds max
                for id in del_tracks:
                    if id < N:
                        del self.tracked_cells_ids[self.tracked_cells[id].cell_id]
                        del self.tracked_cells[id]
                        del assignment[id]
                    else:
                        print("ERROR: id is greater than length of tracker.tracks")
            # Now look for un_assigned detects
            un_assigned_detects = []
            for i in range(len(cell_centers_in_frame)):
                if i not in assignment:
                    un_assigned_detects.append(i)

            # Start new tracker for the cells
            if(len(un_assigned_detects) != 0):
                for i in range(len(un_assigned_detects)):
                    if un_assigned_detects[i] not in self.tracked_cells_ids:    
                        added_cell = self.add_cell(
                            cell_centers_in_frame[un_assigned_detects[i]])
                        self.add_to_frame(
                            cell_centers_in_frame[un_assigned_detects[i]])

            for i in range(len(assignment)):
                self.tracked_cells[i].KF.predict()
                if(assignment[i] != -1):
                    self.tracked_cells[i].skipped_frames = 0
                    self.tracked_cells[i].KF.state, self.tracked_cells[i].prediction = self.tracked_cells[i].KF.correct(
                        np.matrix(cell_centers_in_frame[assignment[i]]).reshape(2, 1), 1)
                else:
                    self.tracked_cells[i].KF.state, self.tracked_cells[i].prediction = self.tracked_cells[i].KF.correct(
                        np.matrix([[0], [1]]).reshape(2, 1), 0)

                if(len(self.tracked_cells[i].positions) > self.max_trace_length_allowed):
                    for j in range(len(self.tracked_cells[i].positions) -
                                   self.max_trace_length_allowed):
                        del self.tracked_cells[i].positions[j]
                self.tracked_cells[i].positions.append(
                    self.tracked_cells[i].prediction[0])
                cv2.putText(image, "id: {}".format(self.tracked_cells[i].cell_id), (int(self.tracked_cells[i].prediction[0, 0]), int(self.tracked_cells[i].prediction[0, 1])), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 20, 40), 2)
        for cell in self.tracked_cells:
            for prediction in cell.positions:
                cv2.circle(image, (int(prediction[0]), int(
                    prediction[1])), 2, (0, 255, 0), -1)

        plt.imshow(image)
        plt.show()

        self.frame_id += 1
        return
