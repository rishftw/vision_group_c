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
        self.cost_thresh_allowed = 20
        self.max_frame_skips_allowed = 20
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

    def get_cell(frame_num, center):
        pass

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
        self.tracks_cell_disappeared_frame.append([])
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
                
        #     # for i in range(N):
        #     #     result = np.where(least_finder[i] < min(least_finder[i])+10)[0]
        #     #     # print(np.where(min(least_finder[i])))
        #     #     # print(self.tracked_cells[i].positions[-1])
        #     #     if assignment[i]!= -1:
        #     #         if(len(result) == 1):
        #     #             if( result[0] == assignment[i]):
        #     #                 pass
        #     #             else:
        #     #                 # assignment[i] = self.tracked_cells[i].cell_id
        #     #                 assignment[i] = -2
        #     #                 self.tracked_cells[i].skipped_frames += 1
        #     #         elif assignment[i] in result:
        #     #             pass
        #     #         else: 
        #     #             # assignment[i]  = self.tracked_cells[i].cell_id
        #     #             assignment[i]  = -2
        #     #             self.tracked_cells[i].skipped_frames += 1
        #     # for i  in range(len(assignment)):
        #     #     print(i,assignment[i])
            
            # Identify tracker with no assignment, if any
            un_assigned_tracks = []
            for i in range(len(assignment)):
                if (assignment[i] != -1):
                    #                     # check for cost distance threshold.
                    #                     # If cost is very high then un_assign (delete) the tracker.track
                    ##print(i, assignment[i], cost[i][assignment[i]])
                    if (cost[i][assignment[i]] > self.cost_thresh_allowed):
                        assignment[i] = -1
                        ## print('unassigning',i,assignment[i])
                        un_assigned_tracks.append(i)
                        self.tracked_cells[i].skipped_frames += 1
                    else:
                        pass

                else:
                    # un_assigned_tracks.append(i)
                    self.tracked_cells[i].skipped_frames += 1
            
            ### print Assignments   
            ## for i in range(len(assignment)):
            ##     print(i,assignment[i])
            
            
            

            # If tracker are not detected for long time, remove them
            del_tracks = []
            for i in range(len(self.tracked_cells)):
                if (self.tracked_cells[i].skipped_frames > 0):
                    del_tracks.append(i)
            
            #check for mitosis
            
            # del_tracks = sorted(del_tracks, reverse=True)
            if len(del_tracks) > 0:  # only when skipped frame exceeds max
                offset = 0
                for id in del_tracks:
                    if id < N:
                        ##print("to delete", id)
                        self.tracks_cell_disappeared_frame[self.frame_id].append(self.tracked_cells[id-offset].cell_id)
                        del self.tracked_cells_ids[id-offset]
                        del self.tracked_cells[id-offset]
                        del assignment[id-offset]
                        offset+=1
                    else:
                        print("ERROR: id is greater than length of tracker.tracks")
            # Now look for un_assigned detects
            un_assigned_detects = []
            for i in range(len(cell_centers_in_frame)):
                ##print("Unassigned check", i, i not in assignment)
                if i not in assignment:
                    un_assigned_detects.append(i)

            # Start new tracker for the cells
            if(len(un_assigned_detects) != 0):
                for i in range(len(un_assigned_detects)):
                    ##print("unassigned detect in tracked_ids?:",i, un_assigned_detects[i] not in self.tracked_cells_ids)
                    # if un_assigned_detects[i] not in self.tracked_cells_ids:    
                    added_cell = self.add_cell(
                        cell_centers_in_frame[un_assigned_detects[i]])
                    self.add_to_frame(
                        cell_centers_in_frame[un_assigned_detects[i]])
            ## for i in range(len(un_assigned_tracks)):
            ##     print(i,un_assigned_tracks[i])
            for i in range(len(self.tracked_cells)):
                cv2.putText(image, "{}".format(self.tracked_cells[i].cell_id), (int(self.tracked_cells[i].prediction[0, 0]), int(self.tracked_cells[i].prediction[0, 1])), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 20, 40), 2)
            for i in range(len(assignment)):
                self.tracked_cells[i].KF.predict()
                if(assignment[i] > -1):
                    self.tracked_cells[i].skipped_frames = 0
                    self.tracked_cells[i].KF.state, self.tracked_cells[i].prediction = self.tracked_cells[i].KF.correct(
                        np.matrix(cell_centers_in_frame[assignment[i]]).reshape(2, 1), 1)
                    # print(self.tracked_cells[i].prediction[0,0])
                    ## FORDEBUGGING
                else:
                    # pass
        #         # elif assignment[i]==-1:
                    self.tracked_cells[i].KF.state, self.tracked_cells[i].prediction = self.tracked_cells[i].KF.correct(
                        np.matrix([[0], [1]]).reshape(2, 1), 0)
        #         else:
        #             # print(type(self.tracked_cells[i].positions[-1]))
        #             self.tracked_cells[i].prediction[0,0] = self.tracked_cells[i].positions[-1][0]
        #             self.tracked_cells[i].prediction[0,1] = self.tracked_cells[i].positions[-1][1]
        #             pass
        #         if(len(self.tracked_cells[i].positions) > self.max_trace_length_allowed):
        #             for j in range(len(self.tracked_cells[i].positions) - self.max_trace_length_allowed):
        #                 del self.tracked_cells[i].positions[j]
                self.tracked_cells[i].positions.append(
                    cell_centers_in_frame[assignment[i]])
        #         # if(self.tracked_cells[i].cell_id == 96):
        #         cv2.putText(image, "{}".format(self.tracked_cells[i].cell_id), (int(self.tracked_cells[i].prediction[0, 0]), int(self.tracked_cells[i].prediction[0, 1])), cv2.FONT_HERSHEY_SIMPLEX,
        #                         0.5, (0, 20, 40), 2)
        # for cell in self.tracked_cells:
        #     for i in range(len(cell.positions)-1):
        #         # cv2.circle(image, (int(prediction[0]), int(
        #         #     prediction[1])), 2, (0, 255, 0), -1)
        #         cv2.line(image, (int(cell.positions[i][0]), int(
        #             cell.positions[i][1])), (int(cell.positions[i+1][0]), int(
        #             cell.positions[i+1][1])), (0, 255, 0), 2)

            ## for i in range(len(self.tracked_cells)):
            ##     if(i==4):
            ##         print(f'self.tracked_cells index: {i}\n,tracked_cells[i].cell_id: {self.tracked_cells[i].cell_id}\n,self.tracked_cells[i].positions :{self.tracked_cells[i].positions}\n')
        # plt.imshow(image,  cmap="gray")
        # plt.show()
        # print('self.tracks_cell_disappeared_frame[self.frame_id]:',self.frame_id,[ i for i in self.tracks_cell_disappeared_frame[self.frame_id]])
        self.frame_id += 1
        return
