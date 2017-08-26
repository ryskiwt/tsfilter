import numpy as np
from numpy import linalg


class Filter(object):
    def __init__(self, F, G, Q, H, R, dtype=np.float64):
        self.dtype = dtype
        self.F = F.astype(dtype).copy()
        self.G = G.astype(dtype).copy()
        self.Q = Q.astype(dtype).copy()
        self.H = H.astype(dtype).copy()
        self.R = R.astype(dtype).copy()
        self.I = np.eye(H.shape[1]).astype(dtype)
        self.initialize()


    def initialize(self, x=None, V=None):
        if x is None:
            self.x = np.zeros([self.H.shape[1],1]).astype(self.dtype)
        else:
            self.x = x.astype(self.dtype).copy()

        if V is None:
            self.V = self.G @ self.Q @ self.G.T
        else:
            self.V = V.astype(self.dtype).copy()


    def get_params(self):
        return self.F, self.G, self.Q, self.H, self.R, self.I

    def get_states(self):
        return self.x, self.V


    def loglikelihood(self, s, hook=None):
        F, G, Q, H, R, I = self.get_params()
        x, V = self.get_states()

        n = s.shape[1]
        A = np.empty(n).astype(self.dtype)
        B = np.empty(n).astype(self.dtype)

        def h(kf, y): pass
        hook = hook if hook is not None else h

        for i in range(n):
            y = s[:,i].reshape(-1,1)
            hook(self, y)

            x[:] = F @ x
            V[:] = F @ V @ F.T + G @ Q @ G.T

            d = H @ V @ H.T + R
            dinv = linalg.inv(d)
            K = V @ H.T @ dinv
            e = y - H @ x
            e[np.isnan(e)] = 0

            if not np.isnan(y).all():
                x[:] = x + K @ e
                V[:] = ( I - K @ H ) @ V

            A[i] = linalg.norm(d)
            B[i] = e.T @ dinv @ e

        return -1/2 * ( n*np.log(2*np.pi) + np.sum( np.log(A) + B ) )


    def filtering(self, s, hook=None):
        F, G, Q, H, R, I = self.get_params()
        x, V = self.get_states()

        n = s.shape[1]
        s2 = np.empty(s.shape).astype(self.dtype)

        def h(kf, y): pass
        hook = hook if hook is not None else h

        for i in range(n):
            y = s[:,i].reshape(-1,1)
            hook(self, y)

            x[:] = F @ x
            V[:] = F @ V @ F.T + G @ Q @ G.T

            if not np.isnan(y).all():
                K = V @ H.T @ linalg.inv( H @ V @ H.T + R )
                e = y - H @ x
                e[np.isnan(e)] = 0
                x[:] = x + K @ e
                V[:] = ( I - K @ H ) @ V

            s2[:,i] = (H @ x).ravel()

        return s2


    def smoothing(self, s, hook_f=None, hook_b=None):
        F, G, Q, H, R, I = self.get_params()
        x, V = self.get_states()

        n = s.shape[1]
        xshape = (x.shape[0], x.shape[1], n)
        Vshape = (V.shape[0], V.shape[1], n)

        x_0 = np.empty(xshape).astype(self.dtype)
        x_1 = np.empty(xshape).astype(self.dtype)
        V_0 = np.empty(Vshape).astype(self.dtype)
        V_1 = np.empty(Vshape).astype(self.dtype)

        def h_f(kf, y): pass
        def h_b(kf): pass
        hook_f = hook_f if hook_f is not None else h_f
        hook_b = hook_b if hook_b is not None else h_b

        for i in range(n):
            y = s[:,i].reshape(-1,1)
            hook_f(self, y)

            x_0[:,:,i] = x
            V_0[:,:,i] = V

            x[:] = F @ x
            V[:] = F @ V @ F.T + G @ Q @ G.T
            x_1[:,:,i] = x
            V_1[:,:,i] = V

            if not np.isnan(y).all():
                K = V @ H.T @ linalg.inv( H @ V @ H.T + R )
                e = y - H @ x
                e[np.isnan(e)] = 0
                x[:] = x + K @ e
                V[:] = ( I - K @ H ) @ V

        s2 = np.empty(s.shape).astype(self.dtype)

        pinv = linalg.pinv

        for i in range(n-1, -1, -1):
            s2[:,i] = (H @ x).ravel()
            hook_b(self)

            A = V_0[:,:,i] @ F.T @ pinv(V_1[:,:,i])
            x[:] = x_0[:,:,i].reshape(-1,1) + A @ (x - x_1[:,:,i].reshape(-1,1))
            V[:] = V_0[:,:,i] + A @ (V - V_1[:,:,i]) @ A.T

        return s2
