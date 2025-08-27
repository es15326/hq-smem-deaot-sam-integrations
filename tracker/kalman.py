import numpy as np



class KalmanFilter:
    def __init__(self):
        # State Transition matrix
        self.A = np.array([[1, 0, 0, 0, 1, 0, 0, 0],  # x1
                           [0, 1, 0, 0, 0, 1, 0, 0],  # y1
                           [0, 0, 1, 0, 0, 0, 1, 0],  # x2
                           [0, 0, 0, 1, 0, 0, 0, 1],  # y2
                           [0, 0, 0, 0, 1, 0, 0, 0],  # vx1
                           [0, 0, 0, 0, 0, 1, 0, 0],  # vy1
                           [0, 0, 0, 0, 0, 0, 1, 0],  # vx2
                           [0, 0, 0, 0, 0, 0, 0, 1]]) # vy2

        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0]])

        # Initial state
        self.x = np.zeros((8, 1))  # [x1, y1, x2, y2, vx1, vy1, vx2, vy2]

        # Initial uncertainty
        self.P = np.eye(8) * 1000

        # Process noise covariance
        self.Q = np.eye(8) * 0.1  # Adjust this depending on the expected movement smoothness

        # Measurement noise covariance
        self.R = np.eye(4) * 5  # Adjust this based on measurement noise

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[:4].flatten()  # return predicted x1, y1, x2, y2

    def update(self, measurement):
        # Kalman Gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))

        # Update the state
        self.x = self.x + np.dot(K, (measurement.reshape(-1, 1) - np.dot(self.H, self.x)))

        # Update uncertainty
        I = np.eye(self.H.shape[1])
        self.P = (I - np.dot(K, self.H)) * self.P