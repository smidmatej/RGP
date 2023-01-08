
 # 
 # This file is part of the RGP distribution (https://github.com/smidmatej/RGP).
 # Copyright (c) 2023 Smid Matej.
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #

import numpy as np
from scipy.linalg import sqrtm



class RBF:
    def __init__(self, L : np.array = np.eye(1), sigma_f : float = 1) -> None:
        """
        Contructor of the RBF function k(x1,x2).
        
        :param: L: Square np.array of dimension d x d. Defines the length scale of the kernel function
        :param: sigma_f: Scalar value used to linearly scale the amplidude of the k(x,x)
        """
        self.L = L
        self.sigma_f = sigma_f
        
    def __call__(self, x1 : np.array, x2 : np.array) -> float:
        """
        Calculate the value of the kernel function given 2 input vectors
        
        :param: x1: np.array of dimension 1 x d
        :param: x2: np.array of dimension 1 x d
        """
        dif = x1-x2
        
        return float(self.sigma_f**2 * np.exp(-1/2*dif.T.dot(np.linalg.inv(self.L*self.L)).dot(dif)))

    def covariance_matrix(self, x1 : np.array, x2 : np.array) -> np.array:
        """
        Fills in a matrix with k(x1[i,:], x2[j,:])
        
        :param: x1: n x d np.array, where n is the number of samples and d is the dimension of the regressor
        :param: x2: n x d np.array, where n is the number of samples and d is the dimension of the regressor
        :param: kernel: Instance of a KernelFunction class
        """
        
        if x1 is None or x2 is None:
            # Dimension zero matrix 
            return np.zeros((0,0))
        
        cov_mat = np.empty((x1.shape[0], x2.shape[0]))*np.NaN
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        
        # for all combinations calculate the kernel
        for i in range(x1.shape[0]):
            a = x1[i,:].reshape(-1,1)
            for j in range(x2.shape[0]):

                b = x2[j,:].reshape(-1,1)

                
                cov_mat[i,j] = self.__call__(a,b)

        return cov_mat

    def __str__(self):
        return f"L = {self.L}, \n\r Sigma_f = {self.sigma_f}"
        

class RGP:
    def __init__(self, X : np.array, y_ : np.array) -> None:
        """
        :param: X: n x dx np.array, where n is the number of basis vectors and dx is the dimension of the regressor
        :param: y_: n x dy np.array, where n is the number of basis vectors and dy is the dimension of the response
        """

        assert X.shape[0] == y_.shape[0], "X and y_ must have the same number of rows"

        if y_.shape[1] > 1:
            raise NotImplementedError("Only 1D response is supported")
        if X.shape[1] > 1:
            raise NotImplementedError("Only 1D regressor is supported")

        self.X = X
        self.y_ = y_
        # Mean function m(x) = 0
        self.K = RBF()
        self.sigma_n = 0.1 # Noise variance
        self.eta = np.array([self.K.L, self.K.sigma_f, self.sigma_n]) # Hyperparameters
        # WARNING: Dont confuse the estimate g at X with the estimate g_t at X_t 
        # p(g_|y_t-1)
        self.mu_g_t_minus_1 = y_ # The a priori mean is the measurement with no y_t 
        self.C_g_t_minus_1 = self.K.covariance_matrix(X, X) + self.sigma_n**2 * np.eye(self.X.shape[0]) # The a priori covariance is the covariance with no y_t


        # Initialize the state
        # p(g_)
        self.mu_g_t = self.mu_g_t_minus_1
        self.C_g_t = self.C_g_t_minus_1 

        # Precompute these since they do not change with regression
        self.K_x = self.K.covariance_matrix(self.X, self.X) + self.sigma_n**2 * np.eye(self.X.shape[0]) # Covariance matrix over X
        self.K_x_inv = np.linalg.inv(self.K_x) # Inverse of the covariance matrix over X
        
        
    def predict(self, X_t_star : np.array, cov : bool, return_Jt : bool = False) -> np.array:
        """
        Predict the value of the response at X_t_star given the data X and y_.
        :param: X_t_star: m x dx np.array, where m is the number of points to predict at and dx is the dimension of the regressor
        :param: cov: Boolean value. If true, the covariance matrix of the prediction is calculated and returned as well
        """

        Jt = self.K.covariance_matrix(X_t_star, self.X).dot(self.K_x_inv) # Gain matrix
        mu_p_t = Jt.dot(self.mu_g_t) # The a posteriori mean of p(g_t|y_t)


        if cov:
            # Calculate and return the covariance matrix too
            B = self.K.covariance_matrix(X_t_star, X_t_star) - Jt.dot(self.K.covariance_matrix(self.X, X_t_star)) # Covariance of p(g_t|g_)
            C_p_t = B + Jt .dot(self.C_g_t).dot(Jt.T) # The a posteriori covariance of p(g_t|y_t)
            #breakpoint()
            if return_Jt:
                return mu_p_t, C_p_t, Jt
            else:
                return mu_p_t, C_p_t
        else:
            if return_Jt:
                return mu_p_t, Jt
            else:
                return mu_p_t


    def regress(self, Xt : np.array, yt : np.array) -> np.array:
        
        # ------ New data received -> step the memory forward ------
        self.mu_g_t_minus_1 = self.mu_g_t # The a priori mean is the estimate of g at X_
        self.C_g_t_minus_1 = self.C_g_t

        
        # ------ Inference step ------
        # Infer the a posteriori distribution of p(g_t|y_t) (the estimate of g_t at X_t)
        mu_p_t, C_p_t, Jt = self.predict(Xt, cov = True, return_Jt = True)

        #breakpoint()

        # ------ Update step ------
        # Update the a posteriori distribution of p(g_|y_t) (the estimate of g at X)
        G_tilde_t = self.C_g_t_minus_1.dot(Jt.T).dot(
                np.linalg.inv(
                    C_p_t + self.sigma_n**2 * np.eye(Xt.shape[0]))) # Kalman gain
        self.mu_g_t = self.mu_g_t_minus_1 + G_tilde_t.dot(yt - mu_p_t) # The a posteriori mean of p(g_|y_t)
        self.C_g_t = self.C_g_t_minus_1 - G_tilde_t.dot(Jt).dot(self.C_g_t_minus_1) # The a posteriori covariance of p(g_|y_t)



        return self.mu_g_t, self.C_g_t

    def learn(self, Xt : np.array, yt : np.array) -> np.array:
        """
        Performs both the updating of the basis vectors, but also the hyperparameter optimization
        """
        
        n_eta = self.eta.shape[0] # State dimension of eta
        n_g = self.mu_g_t.shape[0] # State dimension of g
        n_g_t = yt.shape[0] # State dimension of g_t
        n_p = n_g + n_eta + n_g_t # State dimension of p

        assert n_g_t == 1, "Only one-dimensional regression is supported"
        assert Xt.shape[0] == 1, "Only one-dimensional regression is supported"

        #z_t = np.concatenate((self.y_, self.eta)) # The measurement vector

        # TODO: Do the time update properly
        mu_g_t = self.mu_g_t
        mu_g_t_minus_1 = mu_g_t
        C_g_t = self.C_g_t
        C_g_t_minus_1 = C_g_t

        mu_eta_t = self.eta
        mu_eta_t_minus_1 = mu_eta_t
        # Covariance matrix of eta
        C_eta_t = np.eye(n_eta)
        C_eta_t_minus_1 = C_eta_t

        # Cross-covariance matrix between eta and g
        C_g_eta_t = np.zeros((n_g, n_eta)) 
        C_g_eta_t_minus_1 = C_g_eta_t

        C_z_t = np.bmat([[self.C_g_t, C_g_eta_t_minus_1],[C_g_eta_t_minus_1.T, C_eta_t]])
        C_z_t_minus_1 = C_z_t


        # ------ Inference step ------

        Jt = self.K.covariance_matrix(Xt, self.X).dot(self.K_x_inv) # Gain matrix (same as in regression)
        assert Jt.shape[1] == n_g, "Jt.shape[1] != n_g"
        B = self.K.covariance_matrix(Xt, Xt) - Jt.dot(self.K.covariance_matrix(self.X, Xt)) # Covariance of p(g_t|g_)
        St = C_g_eta_t_minus_1.dot(np.linalg.inv(C_eta_t_minus_1))

        # At is a function of Jt which is a function of eta (nonlinear function)
        At = np.asarray(np.bmat([
                        [np.eye(n_g), np.zeros((n_g, n_eta))],
                        [np.zeros((n_eta, n_g)), np.eye(n_eta)],
                        [Jt, np.zeros((1, n_eta))]])) # I prefer using np arrays instead of np matrices

        mu_w_t = np.zeros((n_p, )) # This is zero because of the zero mean function of GP. Should be nonzero in general
        C_w_t = np.asarray(np.bmat([
            [np.zeros((n_g, n_g)), np.zeros((n_g, n_eta)), np.zeros((n_g, n_g_t))],
            [np.zeros((n_eta, n_g)), np.zeros((n_eta, n_eta)), np.zeros((n_eta, n_g_t))],
            [np.zeros((n_g_t, n_g)), np.zeros((n_g_t, n_eta)), B]]))
        
        assert mu_w_t.shape[0] == At.shape[0], "mu_w_t.shape[0] != At.shape[0]"
        assert mu_w_t.shape[0] == C_w_t.shape[0], "mu_w_t.shape[0] != C_w_t.shape[0]"

        
        # ------ Unscented transform ------
        w, eta_hat = self.__draw_sigma_points(mu_eta_t, C_eta_t)
        s = w.shape[0] # Number of sigma points
        
        mu_p_i = np.empty((s, n_p)) # Allocate memory
        C_p_i = np.empty((s, n_p, n_p)) # Allocate memory
        
        mu_p_t = np.zeros((n_p, )) # Allocate memory
        C_p_t = np.zeros((n_p, n_p)) # Allocate memory
        for i in range(s):
            
            # --------- Individual predictions from sigma points ---------
            # Transform the sigma points
            mu_p_i[i,:] = At.dot(np.concatenate([
                    mu_g_t_minus_1.ravel() + St.dot(eta_hat[i,:] - mu_eta_t_minus_1),
                    eta_hat[i,:]]
                    , axis=0)).ravel() + mu_w_t

            
            tmp_matrix = np.bmat([[C_g_t_minus_1 - St.dot(C_g_eta_t_minus_1.T), np.zeros((n_g, n_eta))],[np.zeros((n_eta, n_g)), np.zeros((n_eta, n_eta))]])
            C_p_i[i,:,:] = At.dot(np.asarray(tmp_matrix)).dot(At.T) + C_w_t
        
            # --------- Combine individual predictions ---------
            # Cummulative sum
            mu_p_t += w[i] * mu_p_i[i,:]
            C_p_t += w[i] * (np.outer(mu_p_i[i,:] - mu_p_t, mu_p_i[i,:] - mu_p_t) + C_p_i[i,:,:])

        breakpoint()
        return mu_p_t, C_p_t

    def __draw_sigma_points(self, mu : np.array, C : np.array) -> np.array:
        """
        Draws sigma points from a Gaussian distribution using the unscented transform
        """
        # --------- Unscented transform ---------

        n = mu.shape[0] # State dimension of mu

        w = np.empty((2*n+1,))
        x = np.empty((2*n+1, n)) # 2n+1 sigma points in R^n
        w[0] = 0.5
        x[0,:] = mu
        

        for i in range(n):
            # index 1 to n
            x[i+1,:] = mu + sqrtm(n/(1-w[0]) * C)[:,i] # ith collumn of the matrix sqrt
            x[i+1+n,:] = mu - sqrtm(n/(1-w[0]) * C)[:,i] # ith collumn of the matrix sqrt
            
            w[i+1] = (1-w[0])/(2*n)
            w[i+1+n] = (1-w[0])/(2*n)
        
        return w, x



