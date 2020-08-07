import numpy as np


class KalmanFilter(object):

    def __init__(self, center, dt=8, sv=6, mv=1):
        """ Initialise Kalman filer """
        super(KalmanFilter, self).__init__()
        self.stateVariance = sv  # E
        self.measurementVariance = mv
        self.dt = dt

        # Vector of observation
        self.b = np.array([[0], [255]])
        # A - state transition matrix
        self.A = np.matrix([[1, self.dt, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, self.dt],  [0, 0, 0, 1]])
        # observation covariance matix- constant throughout the state
        self.H = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0]])  # [0, 0, 1, 0]

        # Error covariance = E*State identity matrix
        self.errorCov = np.matrix(
            self.stateVariance*np.identity(self.A.shape[0]))
        # Observation noise matrix
        self.R = np.matrix(self.measurementVariance*np.identity(
            self.H.shape[0]))
        # Process noise matrix
        self.Q = np.matrix([[self.dt**3/3, self.dt**2/2, 0, 0],
                            [self.dt**2/2, self.dt, 0, 0],
                            [0, 0, self.dt**3/3, self.dt**2/2],
                            [0, 0, self.dt**2/2, self.dt]])
        # Current state of the cell
        self.state = np.matrix([[0], [1], [0], [1]])
        # Predicted state
        self.predictedstate = None

    def predict(self):
        """Predicts the next state of the cell using the 
            previous state information"""
        # X(i) = A*X(i-1)
        self.state = self.A*self.state
        # P(i) = A*P(i-1)*A(Transpose) + Q
        self.predictedErrorCov = self.A*self.errorCov*self.A.T + self.Q
        state_array = np.asarray(self.state)
        self.predictedstate = state_array[0], state_array[2]
        return state_array[0], state_array[2]

    def correct(self, center, flag):
        """Updates the predicted state using the current measurement"""
        if not flag:
            temp = np.asarray(self.state)
            self.b = [temp[0], temp[2]]
        else:
            self.b = center
        # K(i) = P(i)*H(Transpose) * Inverse of ( H*P(i)*H(Transpose) + R )
        self.kalmanGain = self.predictedErrorCov*self.H.T*np.linalg.pinv(
            self.H*self.predictedErrorCov*self.H.T+self.R)
        # X(i) = X(i) + K(i) * (y(i) - H*X(i))
        self.state = self.state + self.kalmanGain * \
            (self.b - (self.H*self.state))
        # P(i) = (I - K(i)*H)* P(i)
        self.erroCov = (np.identity(self.errorCov.shape[0]) -
                        self.kalmanGain*self.H)*self.predictedErrorCov

        state_array = np.asarray(self.state)
        self.predictedstate = state_array[0], state_array[2]
        return self.state, np.array(self.predictedstate).reshape(1, 2)
