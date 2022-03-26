#!/usr/bin/env python3
import numpy as np
import min_jerk as mj
from dmp import *

def min_jerk_trajs(dims=1, n_samples=10, freq=100, duration=1., add_noise=False):
    """
    Generate a set of minimum-jerk trajectories [n_samples x dims x n_length].

    dims   : the dimension of a sample trajectory
    n_samples: the number of trajectories
    freq     : frequency
    duration : time length of trajectories
    add_noise: add gaussian noise
    """
    samples = []
    for i in range(n_samples):

        start = np.zeros(dims)
        goal  = np.ones(dims)
        
        if add_noise:            
            start += np.random.normal(0,0.04,dims)
            goal  += np.random.normal(0,0.04,dims)
       
        _, trj, trj_vel, trj_acc, trj_jerk = mj.min_jerk(start, goal, duration, freq)
        samples.append(trj.T.tolist())

    return np.array(samples)


def problem_1a():
    """
    Train a DMP with a min-jerk traj and reproduce the same traj.
    """    
    n_samples = 1
    dims      = 1
    bfs       = 20
    tau       = 1.
    
    # Data generation
    trajs_demo = min_jerk_trajs(dims=dims, n_samples=n_samples)

    # Learn via DMP original/improved
    dmp = DMPs_discrete(dims=dims, bfs=bfs, tau=tau)
    dmp.learn(trajs_demo)

    # ReProduce a trajectory
    traj, _, _ = dmp.plan()

    # Reproduction w/ visualization
    dmp.plot_traj(trajs_demo, np.expand_dims(traj, axis=0))
    dmp.plot_basis()


def problem_1b():
    """
    Train a DMP with a min-jerk traj and reproduce it with different goals.
    """    
    n_samples = 1
    dims      = 1
    bfs       = 20
    tau       = 1.
    
    # Data generation
    trajs_demo = min_jerk_trajs(dims=dims, n_samples=n_samples)

    # Learn via DMP original/improved
    dmp = DMPs_discrete(dims=dims, bfs=bfs, tau=tau, enable_improved=True)
    dmp.learn(trajs_demo)

    # ReProduce a trajectory
    y0    = None
    trajs = []

    #------------------------------------------------------------
    # Place your code here
    for i in range(10):
        traj, _, _ = dmp.plan(y0, i+1)
        trajs.append(traj)
    #------------------------------------------------------------
    # Reproduction w/ visualization
    dmp.plot_traj(trajs_demo, np.array(trajs))

    
def problem_1c():
    """
    Train DMPs with multiple min-jerk trajectories and reproduce 
    the demo with the same goal.
    """
    n_samples = 10
    dims      = 2
    bfs       = 20
    tau       = 1.
    
    # Data generation
    trajs_demo = min_jerk_trajs(dims=dims, n_samples=n_samples, add_noise=True)

    # Learn via DMP original/improved
    dmp = DMPs_discrete(dims=dims, bfs=bfs, tau=tau)
    dmp.learn(trajs_demo)

    # ReProduce a trajectory
    traj, _, _ = dmp.plan()

    # Reproduction w/ visualization
    dmp.plot_traj(trajs_demo, np.expand_dims(traj, axis=0))

    
def problem_1d():
    """
    Train DMPs with multiple min-jerk trajectories and reproduce 
    the demo with different goals.
    """
    n_samples = 10
    dims      = 2
    bfs       = 20
    tau       = 1.
    
    # Data generation
    trajs_demo = min_jerk_trajs(dims=dims, n_samples=n_samples, add_noise=True)
    print(trajs_demo.shape)
    # Learn via DMP original/improved
    dmp = DMPs_discrete(dims=dims, bfs=bfs, tau=tau, enable_improved=True)
    dmp.learn(trajs_demo)

    # ReProduce a trajectory
    y0    = None
    trajs = []

    #------------------------------------------------------------
    # Place your code here
    for i in range(10):

        traj, _, _ = dmp.plan(y0, (0.8+0.04*i, 0.8+0.04*i))
        trajs.append(traj)
    #------------------------------------------------------------
    # Reproduction w/ visualization
    dmp.plot_traj(trajs_demo, np.array(trajs))
    plot_2d(traj, trajs_demo)

    
    

def problem_1e():
    # Bonus - handwriting
    n_samples = 1000
    dims      = 2
    bfs       = 120
    tau       = 1.
    
    # Data generation
    import lasa_utils
    DataSet = lasa_utils._PyLasaDataSet()
    dataset = DataSet.Trapezoid
    # dataset 종류: 
    # # ['Angle', 'GShape', 'Khamesh', 'LShape', 'Multi_Models_4', 
    # 'Saeghe', 'Spoon', 'WShape', 'BendedLine', 'heee', 'Leaf_1', 
    # 'Multi_Models_1', 'NShape', 'Sharpc', 'Sshape', 'Zshape', 
    # 'CShape', 'JShape_2', 'Leaf_2', 'Multi_Models_2', 'PShape', 
    # 'Sine', 'Trapezoid', 'DoubleBendedLine', 'JShape', 'Line', 
    # 'Multi_Models_3', 'RShape', 'Snake', 'Worm']
    # demos = np.array(dataset_angle.demos)
    print(dataset.demos[0].pos.shape, tau)
    trajs_demo = np.array([dataset.demos[i].pos for i in range(7)])
    print(trajs_demo[0][0][0], trajs_demo[0][0][-1])
    print(trajs_demo.shape)
    
    # Learn via DMP original/improved
    dmp = DMPs_discrete(dims=dims, bfs=bfs, dt=dataset.dt, tau=tau, enable_improved=False)
    dmp.learn(trajs_demo)

    # ReProduce a trajectory
    y0 = None

    traj, _, _ = dmp.plan(None, None)
    #------------------------------------------------------------
    # Reproduction w/ visualization
    plot_2d(traj, trajs_demo)

    # dmp.plot_traj(trajs_demo, np.array(trajs))

    

def plot_2d(traj, trajs_demo):
    # Reproduction w/ visualization
    n_samples, _, _ = trajs_demo.shape
    fig = plt.figure()
    plt.title('Trajectory (X) - Demo (Td) and generated (Tg)')
    for i in range(n_samples):
        plt.plot(trajs_demo[i,0], trajs_demo[i,1], 'r--', label='Td')
    plt.plot(traj[0], traj[1], 'g-', label='Tg')
    plt.legend()
    plt.savefig('plot2d.png')
    # plt.show()        


if __name__ == "__main__":
    import optparse
    p = optparse.OptionParser()
    p.add_option('--subproblem', '-s', action='store',
                 default='a', help='use this option to specify the subproblem')
    opt, args = p.parse_args()
    


    if opt.subproblem == 'a':
        # Train DMP with a 1-dimensional single demo and reproduce it (also plot basis functions)
        print("Problem 1A")
        problem_1a()
    elif opt.subproblem == 'b':
        # Adapt the DMP to other goals
        print("Problem 1B")
        problem_1b()
    elif opt.subproblem == 'c':
        # Train DMP with multiple demos and reproduce it 
        print("Problem 1C")
        problem_1c()
    elif opt.subproblem == 'd':
        # Adapt the DMP to other goals    
        print("Problem 1D")
        problem_1d()
    elif opt.subproblem == 'e':
        # Adapt the DMP to other goals    
        print("Problem 1E")
        problem_1e()
    else:
        print("Error! Please specify the sub problem!")




    
    
