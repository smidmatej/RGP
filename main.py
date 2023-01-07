import numpy as np



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
    def __init__(self, X : np.array, g_ : np.array) -> None:
        """
        :param: X: n x dx np.array, where n is the number of basis vectors and dx is the dimension of the regressor
        :param: g_: n x dy np.array, where n is the number of basis vectors and dy is the dimension of the response
        """

        assert X.shape[0] == g_.shape[0], "X and g_ must have the same number of rows"

        self.X = X
        self.g_ = g_
        self.K = RBF()

        # WARNING: Dont confuse the estimate g_ at X with the estimate g_t at X_t 
        # p(g_|y_t-1)
        self.mu_g_t_minus_1 = g_ # The a priori mean is the measurement with no y_t 

    def __call__(self, Xt : np.array):
        # Mean function m(x) = 0
        Jt = self.K.covariance_matrix(Xt, self.X) * np.linalg.inv(self.K.covariance_matrix(self.X, self.X))
        mu_p_t = Jt.dot(self.mu_g_t_minus_1) # The a posteriori mean of p(g_t|y_t)


        return

