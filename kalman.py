# https://github.com/zziz/kalman-filter

class KalmanFilter(object):
    def __init__(self, 
                 F, # dynamic model [function]
                 dF, # derivative of dynamic [function]
                 H, # measurement model [constant]
                 Q, # covariance of process noise [constant, assume iid]
                 R, # covariance measurement noise [constant, assume iid]
                 P, # 
                 h0):

        self.F = F
        self.dF = dF
        self.H = H
        self.Q = Q 
        self.R = R 
        self.P = P 
        self.h = h0 
    
    def predict(self, u = 0):
        self.h = self.F(self.h) # + np.dot(self.B, u)

        self.P = self.dF(self.h) * self.P * self.dF(self.h) + self.Q

    def update(self, x): # x: measurement
        y = x - self.H * self.h # helper

        S = self.R + self.H * self.P * self.H # helper for Kalman gain

        K = self.P * self.H / S # Kalman gain
        self.h = self.h + K * y # y is the disagrement 

        # I = np.eye(self.n)
        _quad_ = (1 - K * self.H)  
        self.P = _quad_ * self.P * _quad_ + K * self.R * K

        return self.h
