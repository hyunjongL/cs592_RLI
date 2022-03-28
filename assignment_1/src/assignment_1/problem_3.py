#!/usr/bin/env python3
import os
import numpy as np
import json
import matplotlib.pylab as plt
import rospy
from scipy.signal import butter, lfilter, freqz, lfilter_zi

import misc
from dmp import *
import joint_trajectory_client as jtc
from trajectory_msgs.msg import *
## import baxter_interface
## from baxter_interface import CHECK_VERSION

from pykdl_utils.kdl_kinematics import create_kdl_kin
from trac_ik_python.trac_ik import IK  
from hrl_geom.pose_converter import PoseConv

# https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    zi = lfilter_zi(b, a) * data[0]
    y, _ = lfilter(b, a, data, zi=zi)
    return y


def extract_data():
    """
    Extract a pouring motion demo from the MIME dataset. (you can download other demo. if you want.)
    """    
    cur_dir = os.getcwd().split('/')[-1]
    assert cur_dir=='assignment_1', "Run the program on the assignment_1 folder. Current directory is {}".format(cur_dir)

    if os.path.isdir('dataset') is False:
        os.mkdir('dataset')
    os.chdir( 'dataset' )
                
    url = 'https://www.dropbox.com/sh/wmyek0jhrpm0hmh/AADAO2L1qN5BwOBthyMG82ima/4315Aug02?dl=0.zip'
    if os.path.isfile(url.split('/')[-1]) is False:
        os.system('wget '+url)

    if os.path.isfile('joint_angles.txt') is False:
        os.system('unzip 4315Aug02?dl=0.zip')

    print (os.getcwd())
    data = []
    for line in open('joint_angles.txt', 'r'):
        data.append( json.loads(line))
    
    # Get left/right arm trajectories
    joint_names = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']

    l_arm_traj = []
    r_arm_traj = []
    for d in data:
        v = []
        for name in joint_names:
            v.append(d['left_'+name])
        l_arm_traj.append(v)

        v = []
        for name in joint_names:
            v.append(d['right_'+name])
        r_arm_traj.append(v)
        
    os.chdir( '..' )
    return np.swapaxes(r_arm_traj, 0, 1)


def fk_request(fk_solver, joint_angle):
    # https://github.com/ros/kdl_parser/issues/44
    # https://github.com/ros/kdl_parser/commit/9c6c7c60f27245d5a7ea9c38ec6f96ac6557c3b2
    # 라이브러리 내에서 오류를 발생하는 None 에 해당하는 부분 코드를 fixed 로 수정함.
    '''
    Forward kinematics that returns a desired pose
        
    fk_solver object: forward kinematics from KDL.
    joint_angle list/array: a list of joint target joint angles
    '''
    homo_mat = fk_solver.forward(joint_angle)
    pos, quat = PoseConv.to_pos_quat(homo_mat)
    return pos, quat


def ik_request(ik_solver, poses, seed_angle):
    '''
    Inverse kinematics that returns a sequence of desired joint angles
        
    ik_solver object: inverse kinematics from TRAC_IK.
    poses Pose: a sequence of target poses
    seed_angle list/array: reference angles (for nullspace control)
    '''

    if type(poses) is not list:
        poses = [poses]

    joint_positions = []
    ers=0
    for i, ps in enumerate(poses):
        if i%10==0: print (i, len(poses))
        ret = ik_solver.get_ik(seed_angle,
                                   ps.position.x,    
                                   ps.position.y,    
                                   ps.position.z,
                                   ps.orientation.x,
                                   ps.orientation.y,
                                   ps.orientation.z,
                                   ps.orientation.w,
                                   bx=1e-4, by=1e-4, bz=1e-4,
                                   brx=1e-4, bry=1e-4, brz=1e-4)
        
        
        if ret is None:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found in {}th pose.".format(i))
            ers+=1
            continue
            # return False
        seed_angle = ret

        joint_positions.append(ret)
    print('errors:', ers)
    return np.swapaxes(joint_positions, 0,1)



def plot_traj(trajs, trajs_demo=None):
    """
    """
    
    fig = plt.figure()

    for i, traj in enumerate(trajs):
        fig.add_subplot(len(trajs), 1, i+1)
        plt.plot(traj, label=str(i))

        if trajs_demo is not None:
            plt.plot(trajs_demo[0][i], 'r-', label=str(i))
    plt.savefig('plot2d.png')
    # plt.show()


def problem_3a1(enable_plot=True):
    traj = extract_data()

    dims      = len(traj)
    bfs       = 30
    tau       = 1.
    freq      = 100
    duration  = 1.
    dt        = 1./freq

    traj_demo = np.expand_dims(traj, axis=0)

    # Learn via DMP original/improved
    dmp = DMPs_discrete(dims=dims, bfs=bfs, tau=tau, dt=dt,
                            enable_improved=True)
    traj, _, _ = dmp.learn(traj_demo)
    traj, _, _ = dmp.plan()

    if enable_plot: plot_traj(traj, traj_demo)
    return traj
    

def problem_3a2(limb='right'):
    # https://cobots.tistory.com/5
    # 로봇에게 익숙한 "Joint space" 와 사람에게 익숙한 "Cartesian space" 간 변환하여 가르치기
    # Joint space의 coordinates 를 forward kinematics 로 cartesian으로 변환하여 그것을 dmp로 학습
    # dmp plan은 그러면 cartesian space에서 smooth route를 만들 것임.
    # 그것 planned trajectory를 inverse kinematics 로 변환한 joint space의 좌표를 로봇에 전달.

    dt = 0.01
    
    print("Initializing node... ")
    rospy.init_node("rsdk_joint_trajectory_client_%s" % (limb,))
    
    #r_traj_data = extract_data()
    r_traj_data = problem_3a1(enable_plot=False)
    # print (jtc.get_joint_names(limb))
    # ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']
    print(r_traj_data.shape)
    # import pykdl_utils
    # fk_request(pykdl_utils, r_traj_data)
    # exit()

    # Command Current Joint Positions first
    points = []
    current_angles = jtc.get_current_angles(limb)
    point = JointTrajectoryPoint()
    point.positions = current_angles
    point.velocities = [0]*7
    point.time_from_start = rospy.Duration.from_sec(0.0)
    points.append(point)

    point = JointTrajectoryPoint()
    point.positions = r_traj_data[:,0]
    point.velocities = [0]*7
    point.time_from_start = rospy.Duration.from_sec(4.0)
    points.append(point)
    print("Waiting at the initial posture")
    jtc.move(limb, points)
    # exit()
    t = 0.
    points = []    
    #------------------------------------------------------------
    # Place your code here
    print(r_traj_data[0].shape)
    for i in range(len(r_traj_data[0])):

        new_point = JointTrajectoryPoint()
        new_point.positions = r_traj_data[:, i]
        new_point.velocities = (new_point.positions - point.positions) / dt
        point.time_from_start = rospy.Duration.from_sec(4.0 + dt * (i+1))
        points.append(point)
        point = new_point

        
    #points.append(point)
    #------------------------------------------------------------

    # Run the trajectory
    jtc.move(limb, points)
    print("Exiting - Joint Trajectory Action Complete")

        
def problem_3bc(limb='right', goal=None):
    """
    Check if you want better orientation-based DMP: 
    Ude et al. Orientation in cartesian space dynamic movement primitives. In IEEE International Conference on Robotics and Automation (ICRA), 2014.
    """
    dims      = 7
    bfs       = 30
    tau       = 1.
    freq      = 100
    duration  = 1.
    dt        = 1./freq
    sample_scale =  20
    r_traj_data  = extract_data()
    n_samples = r_traj_data.shape[1]
    # 7x2234
    # FK, IK solvers for Baxter
    fk_solver = create_kdl_kin('base', 'right_gripper')
    ik_solver = IK('base', 'right_gripper', timeout=0.025, epsilon=1e-3, solve_type="Distance")
    # pos, quat = fk_request(fk_solver, r_traj_data.T[0])
    # print(pos, quat)

    #------------------------------------------------------------
    # Place your code here
    # make a list of x,y,z,qx,qy,qz,qw
    # [x, y, z, qx, qy, qz, qw] = 𝐹𝐾(𝜃𝑠0, 𝜃𝑠1, 𝜃𝑒0, 𝜃𝑒1, 𝜃𝑤0, 𝜃𝑤1, 𝜃𝑤2)
    pose_list = []
    traj_demo = []
    for i in range(r_traj_data.shape[1]):
        pos, quat = fk_request(fk_solver, r_traj_data[:,i])
        pose_list.append(pos)
        pos.extend(quat)
        traj_demo.append(pos)
    traj_demo = np.array(traj_demo).T
    traj_demo = np.reshape(traj_demo, (1, dims, n_samples))

    #------------------------------------------------------------

    # Learn via DMP original/improved
    dmp = DMPs_discrete(dims=dims, bfs=bfs, tau=tau, dt=dt,
                            enable_improved=True)
    traj, _, _ = dmp.learn( traj_demo ) #[:,:6,:] )
    # setting a goal
    traj, _, _ = dmp.plan(goal=goal)


    order = 6
    cutoff = 10  # desired cutoff frequency of the filter, Hz

    for i in range(traj.shape[0]):
        traj[i,:] = butter_lowpass_filter(traj[i,:], cutoff, freq, order)



    # normalize the quaternions
    traj[3:] /= (np.sum( traj[3:]**2, axis=0) - 0.005) # np.sqrt(np.sum( traj[3:]**2, axis=0))
    # Python 
    # np.set_printoptions(threshold=sys.maxsize)
    # print(np.sum(traj[3:]**2, axis=0))
    # exit()

    #--------------# 
    # Demonstrate the use of the filter.
    # First make some data to be filtered.

    # Filter the data, and plot both the original and filtered signals.
    order = 5
    cutoff = 10  # desired cutoff frequency of the filter, Hz

    # for i in range(traj.shape[0]):
    #     traj[i,:] = butter_lowpass_filter(traj[i,:], cutoff, freq, order)

    # plot_traj(traj, traj_demo)
    #--------------# 

    # print(np.std(traj - traj_demo[0]), np.mean(traj - traj_demo[0]), np.max(traj - traj_demo[0]), np.min(traj - traj_demo[0]))
    # exit()
    # conver the pos+quaternion trajectory to pose list
    pose_list = []
    for i in range(len(traj[0])):
        # reduce the number of samples (OPTION)
        if i%sample_scale==0:
            ps = misc.list2Pose(traj[:,i])
            pose_list.append( ps )
        
    print("Initializing node... ")
    rospy.init_node("rsdk_joint_trajectory_client_%s" % (limb,))


    
    # Inverse Kinematics
    current_angles = jtc.get_current_angles(limb)
    for i in range(10):
        joint_des_traj = ik_request(ik_solver, pose_list, current_angles)
        if joint_des_traj is not False: break
    np.set_printoptions(threshold=sys.maxsize)
    print(joint_des_traj.T)
    plot_traj(joint_des_traj, [joint_des_traj])
    # print(len(pose_list), joint_des_traj.shape)
    # exit()

    # Command Current Joint Positions first
    points = []
    current_angles = jtc.get_current_angles(limb)
    point = JointTrajectoryPoint()
    point.positions = current_angles
    point.velocities = [0]*7
    point.time_from_start = rospy.Duration.from_sec(0.)
    points.append(point)

    # Trajectory의 시작으로 planned trajectory의 시작을 사용. (기존에는 training data의 시작이었음.)
    # 움직임 시작하자마자 튀는 현상을 잡기 위함임.
    point = JointTrajectoryPoint()
    point.positions = joint_des_traj[:,0]
    point.velocities = [0]*7
    point.time_from_start = rospy.Duration.from_sec(4.0)
    points.append(point)    
    jtc.move(limb, points)
    
    print("set position done")

    #-------
    if joint_des_traj is False:
        rospy.logerr("Maybe unreachable goal pose... ")
        sys.exit()
    t = 0.
    points = []

    # input("Press <Enter> to Continue...")
    #------------------------------------------------------------
    # Place your code here

    # Get the current angles
    current_angles = jtc.get_current_angles(limb)
    
    point = JointTrajectoryPoint()
    point.positions = current_angles
    point.velocities = np.array([0]*7)
    point.time_from_start = rospy.Duration.from_sec(0.)
    points.append(point)
    # Smoothing the output joint trajectory

    order = 5
    cutoff = 4
    for i in range(joint_des_traj.shape[0]):
        joint_des_traj[i,:] = butter_lowpass_filter(joint_des_traj[i,:], cutoff/sample_scale, freq/sample_scale, order)

    # Create the points

    for i in range(joint_des_traj.shape[1] - 1):
        newpoint = JointTrajectoryPoint()
        newpoint.positions = joint_des_traj[:,i+1]
        newpoint.velocities = (newpoint.positions - point.positions) / (dt*sample_scale)
        newpoint.time_from_start = rospy.Duration.from_sec(dt * (i + 1) * sample_scale)
        points.append(newpoint)
        point = newpoint



    
    #------------------------------------------------------------
    
    # Run the trajectory
    jtc.move(limb, points)
    print("Exiting - Joint Trajectory Action Test Complete")
        
    
if __name__ == "__main__":
    import optparse
    p = optparse.OptionParser()
    p.add_option('--subproblem', '-s', action='store',
                 default='a1', help='use this option to specify the subproblem')
    opt, args = p.parse_args()

    if opt.subproblem == 'a1':
        # Obtain data
        # Train a DMP with high-dimensional joint-space data and reproduce it
        print("Problem 3A-1")
        problem_3a1()
    
    elif opt.subproblem == 'a2':
        # Reproduce it with a Baxter robot
        print("Problem 3A-2")
        problem_3a2()

    elif opt.subproblem == 'b':
        # Train a DMP with converted Cartesian-space data and reproduce it
        # Reproduce it with a Baxter robot
        print("Problem 3B")
        problem_3bc()
    
    elif opt.subproblem == 'c':
        # Train a DMP with converted Cartesian-space data and reproduce it
        # Reproduce it with a Baxter robot
        print("Problem 3C")
        # Adapt the goal to multiple poses and reproduce it with the Baxter robot

        #------------------------------------------------------------
        # Place your code here. Following is an example
        goal = [ 0.5319781 , -0.37729777,  0.04209853,  0.62513343,  0.46658677, 0.46685764, -0.41659203]
        #------------------------------------------------------------
        
        problem_3bc(goal=goal)
