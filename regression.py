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
 
# Core
import numpy as np
from RGP import RBF, RGP

# Plotting
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec
from matplotlib.collections import PolyCollection

# Cosmetic
import seaborn as sns
from tqdm import tqdm




def main():

    # ----- The basis vectors -----
    X_ = np.arange(-10,10,1).reshape(-1,1)
    y_ = np.random.normal(0, 0, size=X_.shape)

    # ----- The true function we are trying to approximate -----
    X_query = np.arange(-10,10,0.2).reshape(-1,1)
    y_true = np.sin(X_query)

    # ----- The training data -----
    n_training = 50
    X_t = np.random.uniform(-10,10, size=(n_training,1))
    X_t = np.sort(X_t, axis=0)

    # Put half of the data at the end in reverse order
    X_t = np.concatenate((X_t[::2], np.flip(X_t[::2])), axis=0)
    sample_noise = 0.1
    y_t = np.sin(X_t) + np.random.normal(0, sample_noise, size=X_t.shape)


    rgp = RGP(X_, y_)
    mean_training = [None] * (X_t.shape[0]+1)
    cov_training = [None] * (X_t.shape[0]+1)
    g_ = [None] * (X_t.shape[0]+1)

    mean_training[0], cov_training[0] = rgp.predict(X_query, cov=True)
    g_[0] = rgp.mu_g_t # Current estimate of g at X

    print('Training model recursively...')
    pbar = tqdm(total=X_t.shape[0])
    for t in range(X_t.shape[0]):
        rgp.regress(X_t[t,:], y_t[t,:])
        mean_training[t+1], cov_training[t+1] = rgp.predict(X_query, cov=True)
        g_[t+1] = rgp.mu_g_t # Current estimate of g at X
        pbar.update()
    pbar.close()


    # ----------------- PLOT -----------------


    cs = [[x/256 for x in (205, 70, 49)], \
            [x/256 for x in (105, 220, 158)], \
            [x/256 for x in (102, 16, 242)], \
            [x/256 for x in (7, 59, 58)]]

    # Color scheme convert from [0,255] to [0,1]
    cs = [[x/256 for x in (8, 65, 92)], \
            [x/256 for x in (204, 41, 54)], \
            [x/256 for x in (118, 148, 159)], \
            [x/256 for x in (232, 197, 71)]] 



    plt.style.use('fast')
    sns.set_style("whitegrid")

    

    fig = plt.figure(figsize=(10,10), dpi=100)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax = [None]*4

    ax[0] = fig.add_subplot(gs[0, 0])
    ax[1] = fig.add_subplot(gs[0, 1])
    ax[2] = fig.add_subplot(gs[1, 0])
    ax[3] = fig.add_subplot(gs[1, 1])

        
    ax[0].plot(X_query, y_true, color=cs[1], label='True function')

    ax[0].plot(X_query, mean_training[0], '--', color=cs[0], label='E[g(x)]')
    ax[0].scatter(X_, y_, marker='o', s=20, color=cs[0], label='Basis Vectors')
    
    
   
    ax[0].fill_between(X_query.reshape(-1),
        mean_training[0].reshape(-1) - 2*np.sqrt(np.diag(cov_training[0])), 
        mean_training[0].reshape(-1) + 2*np.sqrt(np.diag(cov_training[0])), color=cs[3], alpha=0.1)


    ax[0].set_xlim((min((min(X_t), min(X_query), min(X_))) , max((max(X_t), max(X_query), max(X_)))))
    #plt.set_ylim((min((min(y_), min(y_t), min(y_true))) , max((max(y_), max(y_t), max(y_true)))))
    ax[0].set_ylim((-2,2))
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend()
    ax[0].set_title('Prior with no training data')


    ax[1].plot(X_query, y_true, color=cs[1], label='True function')
    ax[1].scatter(X_, g_[-1], marker='o', s=20, color=cs[0], label='Basis Vectors')
    ax[1].scatter(X_t, y_t, marker='+', color=cs[2], label='Training samples')
    ax[1].plot(X_query, mean_training[-1], '--', color=cs[0], label='E[g(x)]')
    
    ax[1].fill_between(X_query.reshape(-1),
        mean_training[-1].reshape(-1) - 2*np.sqrt(np.diag(cov_training[-1])), 
        mean_training[-1].reshape(-1) + 2*np.sqrt(np.diag(cov_training[-1])), color=cs[3], alpha=0.1)



    ax[1].set_xlim((min((min(X_t), min(X_query), min(X_))) , max((max(X_t), max(X_query), max(X_)))))
    #plt.set_ylim((min((min(y_), min(y_t), min(y_true))) , max((max(y_), max(y_t), max(y_true)))))
    ax[1].set_ylim((-2,2))
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].legend()
    ax[1].set_title('Posterior after training')


    ax[2].plot(X_query, y_true - mean_training[0], color=cs[0])
    ax[2].set_xlim((min((min(X_t), min(X_query), min(X_))) , max((max(X_t), max(X_query), max(X_)))))
    #plt.set_ylim((min((min(y_), min(y_t), min(y_true))) , max((max(y_), max(y_t), max(y_true)))))
    ax[2].set_ylim((-2,2))
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].set_title('Difference (true - prior)')

    ax[3].plot(X_query, y_true - mean_training[-1], color=cs[0])
    ax[3].set_xlim((min((min(X_t), min(X_query), min(X_))) , max((max(X_t), max(X_query), max(X_)))))
    #plt.set_ylim((min((min(y_), min(y_t), min(y_true))) , max((max(y_), max(y_t), max(y_true)))))
    ax[3].set_ylim((-2,2))
    ax[3].set_xlabel('x')
    ax[3].set_ylabel('y')
    ax[3].set_title('Difference (true - prior)')

    plt.savefig('img/regression_comparisson.png')
    plt.show()


    # ----------------- ANIMATION -----------------
    print('Rendering animation...')

    def animate(i):

        line_mean.set_data(X_query, mean_training[i])

        scat_basis_vectors.set_offsets(np.array([X_.ravel(), g_[i].ravel()]).T)
        scat_training_points.set_offsets(np.array([X_t[:i,:].ravel(), y_t[:i,:].ravel()]).T)

        global fill_between
        fill_between.remove()

        fill_between = ax.fill_between(X_query.reshape(-1),
            mean_training[i].reshape(-1) - 2*np.sqrt(np.diag(cov_training[i])), 
            mean_training[i].reshape(-1) + 2*np.sqrt(np.diag(cov_training[i])), color=cs[1], alpha=0.2)
        


        pbar.update()


    animation.writer = animation.writers['ffmpeg']
    plt.ioff() # Turn off interactive mode to hide rendering animations




    plt.style.use('fast')
    sns.set_style("whitegrid")

    #gs = gridspec.GridSpec(2, 2)
    
    fig = plt.figure(figsize=(10,10), dpi=60)
    ax = fig.add_subplot(111)

    line_mean, = ax.plot([], [], '--', color=cs[0], label='E[g(x)]')
    scat_training_points = ax.scatter([], [], marker='+', color=cs[1], label='Training samples')
    scat_basis_vectors = ax.scatter([], [], marker='o', color=cs[2], label='Basis Vectors')

    # Hack to be able to change the fill at each step
    global fill_between
    fill_between = ax.fill_between([],
        [], 
        [], color=cs[3], alpha=0.2)



    ax.plot(X_query, y_true, color='gray')

    ax.set_xlim((min((min(X_t), min(X_query), min(X_))) , max((max(X_t), max(X_query), max(X_)))))
    #ax.set_ylim((min((min(y_), min(y_t), min(y_true))) , max((max(y_), max(y_t), max(y_true)))))
    ax.set_ylim((-2,2))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Recursive Gaussian Process')
    ax.legend()




    pbar = tqdm(total=X_t.shape[0])
    number_of_frames = len(mean_training)
    ani = animation.FuncAnimation(fig, animate, frames=number_of_frames, interval=500)     
    ani.save('img/regression.gif', writer='imagemagick', fps=10, dpi=30)
    ani.save('img/regression.mp4', writer='ffmpeg', fps=10, dpi=100)
    pbar.close()
    plt.show()


    
    
    
    
    


if __name__ == '__main__':
    main()