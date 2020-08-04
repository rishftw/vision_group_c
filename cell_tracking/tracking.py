from collections import OrderedDict
import numpy as np
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from collections import deque
import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import imutils
from cell_tracking.kalman import KalmanFilter
class Cell():
    def __int__(self):
        self.cell_id = None
        self.predictedState = None

    def addPath(self, center, cell_id):
        self.positions = deque()
        self.KF = KalmanFilter(center, 8, 0.5, 1)
        self.missedFrame = 0
        self.predictedState = np.asarray(center).reshape(1, 2)
        self.cell_id = cell_id




class Tracker():
    def __init__(self, bias, max_missedFrame):
        self.cell_id = 0
        self.bias = bias
        self.max_missedFrame = max_missedFrame
        self.cells = []
        self.current_cells = []

    def addCell(self, center, cell_id):
        pathObj = Cell()
        pathObj.addPath(center, cell_id)
        self.cells.append(pathObj)
        self.current_cells.append(cell_id)

    def removeCell(self, cell_id):
        del self.cells[cell_id]

    def track(self, centersList):
        centers = np.zeros((len(centersList), 2), dtype="int")
        # Assinging individual trackers to the cells
        for (i, center) in enumerate(centersList):
            centers[i] = center
        if len(self.cells) == 0:
            for c in range(0, len(centers)):
                self.addCell(centers[c], c)

        # Updating the Frobenius cost norm
        cost = np.zeros(shape=(len(self.cells), len(centers)))
        least_finder = np.zeros(shape=(len(self.cells), len(centers)))
        for i in range(len(self.cells)):
            for j in range(len(centers)):
                try:
                    temp = self.cells[i].predictedState - centers[j]
                    # Fnorm = np.linalg.norm(temp, axis=1,ord= 2)
                    # cost[i][j] = Fnorm
                    diff = self.cells[i].prediction - \
                            centers[j]
                    distance = np.sqrt(
                            diff[0][0]*diff[0][0] + diff[0][1]*diff[0][1])
                    cost[i][j] = distance
                    least_finder[i][j] = distance
                except:
                    pass

        # Applying the Hungarian min weight matching algorithm
        cost = cost * 0.5
        row, col = linear_sum_assignment(cost)

        # Updating the min-weight matched pairs
        assigned_cells = [-1]*len(self.cells)
        for i in range(len(row)):
            assigned_cells[row[i]] = col[i]

        assigned_cells_unassigned = []
        for i in range(len(assigned_cells)):
            if assigned_cells[i] != -1:
                if cost[i][assigned_cells[i]] > self.bias:
                    assigned_cells[i] = -1
                    assigned_cells_unassigned.append(i)
            else:
                self.cells[i].missedFrame += 1

        for i in range(len(self.cells)):
                result = np.where(least_finder[i] < min(least_finder[i])+10)[0]
                # print(np.where(min(least_finder[i])))
                # print(self.tracked_cells[i].positions[-1])
                if assigned_cells[i]!= -1:
                    if(len(result) == 1):
                        if( result[0] == assigned_cells[i]):
                            pass
                        else:
                            # assigned_cells[i] = self.tracked_cells[i].cell_id
                            assigned_cells[i] = -1
                            assigned_cells_unassigned.append(i)
                            self.cells[i].missedFrame += 1
                    elif assigned_cells[i] in result:
                        pass
                    else: 
                        # assigned_cells[i]  = self.tracked_cells[i].cell_id
                        assigned_cells[i]  = -1
                        assigned_cells_unassigned.append(i)
                        self.cells[i].missedFrame += 1

        missing_elements = []
        for i in range(len(self.cells)):
            if self.cells[i].missedFrame > self.max_missedFrame:
                missing_elements.append(i)

        missing_elements = sorted(missing_elements, reverse=True)
        if len(missing_elements) > 0:
            for i in range(len(missing_elements)):
                self.removeCell(missing_elements[i])
                del assigned_cells[missing_elements[i]]
                del self.current_cells[missing_elements[i]]

        un_assigned_cells = []
        for i in range(len(centers)):
            if i not in assigned_cells:
                un_assigned_cells.append(i)

        if len(un_assigned_cells) != 0:
            for i in range(len(un_assigned_cells)):
                if un_assigned_cells[i] not in self.current_cells:
                    self.addCell(centers[un_assigned_cells[i]], un_assigned_cells[i])


        for i in range(len(assigned_cells)):
            c = np.matrix([[0], [1]])
            self.cells[i].KF.predict()
            if (assigned_cells[i] != -1):
                self.cells[i].missedFrame = 0
                self.cells[i].KF.state, self.cells[i].predictedState = self.cells[i].KF.correct(
                    np.matrix(centers[assigned_cells[i]]).reshape(2, 1), 1)
            else:
                self.cells[i].KF.state, self.cells[i].predictedState = self.cells[i].KF.correct(
                    np.matrix(c).reshape(2, 1), 0)
            self.cells[i].positions.append(
                self.cells[i].predictedState)
