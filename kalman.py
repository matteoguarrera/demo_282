# https://github.com/zziz/kalman-filter

class KalmanFilter(object):
    def __init__(self, 
                 F, # dynamic model [function]
                 dF, # derivative of dynamic [function]
                 H, # measurement model [constant]
                 Q, # covariance of process noise [constant, assume iid]
                 R, # covariance measurement noise [constant, assume iid]
                 P, # 
                 x0):

        self.F = F
        self.dF = dF
        self.H = H
        self.Q = Q 
        self.R = R 
        self.P = P 
        self.x = x0 
    
    def predict(self, u = 0):
        self.x = self.F(self.x) # + np.dot(self.B, u)

        self.P = self.dF(self.x) * self.P * self.dF(self.x) + self.Q

    def update(self, z): # z: measurement
        y = z - self.H * self.x # helper

        S = self.R + self.H * self.P * self.H # helper for Kalman gain

        K = self.P * self.H /S # Kalman gain
        self.x = self.x + K * y # y is the disagrement 

        # I = np.eye(self.n)
        _quad_ = (1 - K * self.H)  
        self.P = _quad_ * self.P * _quad_ + K * self.R * K

        return self.x
