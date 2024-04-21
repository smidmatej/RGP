# Recursive Gaussian Process
This repository implements the Recursive Gaussian Process algorithm as described in [[1]](#1).

After choosing a set of m basis vectors, we can use the algorithm to update their mean and covariance recursively. 
All the information about the regressor is stored in the basis vector estimate and its covariance. 
We can use this algorithm to learn on a large amount of data n, while keeping the memory requirements roughly $O(m + m^2)$ (mean + covariance matrix). 
The computational requirements are also vastly reduced.


## Example
Here we are trying to estimate a sine function $\text{sin}(x)$. We draw samples from the function as $y=\text{sin}(x)+e$, where $e$ is drawn from a normal distribution with zero mean and 0.1 variance. The RGP is initialized with basis vectors uniformly distributed on $\left<-10,10\right>$ with their $y$ values set to zero. During training, the samples are fed to the RGP which updates the $y$ values at the basis vectors iteratively. The algorithm is able to provide smooth, differentiable estimate with variance along the sampled interval, while keeping the memory requirements minimal: 20 $x$ floats, 20 $y$ floats and 3 hyperparameters. The RGP hyperparameters are initialized to [1,1,1] and stay constant during the training process.
![animated_regression](/tree/master/img/regression.gif)
![before_after](/img/regression_comparisson.png)
<img src="/img/regression_comparisson.png" width="150" height="100">
## References
<a id="1">[1]</a> 
Huber, M. F. (2014). Recursive Gaussian process: On-line regression and learning. Pattern Recognition Letters, 45, 85-91.
