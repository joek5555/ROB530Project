import numpy as np
import os
import matplotlib.pyplot as plt

from PF import particle_filter



landmark_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Landmark_Groundtruth.dat'))
robot1_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Groundtruth.dat'))
robot1_measurement_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Measurement.dat'))
robot1_odometry_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Odometry.dat'))

landmark_groundtruth = np.loadtxt(landmark_groundtruth_path)
robot1_groundtruth = np.loadtxt(robot1_groundtruth_path)
robot1_measurments = np.loadtxt(robot1_measurement_path)
robot1_odometry = np.loadtxt(robot1_odometry_path)

class system:
    def __init__(self):
        self.process_model = None
        self.process_covariance = None
        self.process_input_size = None

        self.measurement_model = None
        self.measurement_covariance = None

def process_model(x,u):

    output = np.zeros(3)
    output[0] = x[0] + (-u[0] / u[1] * np.sin(x[2]) + u[0] / u[1] * np.sin(x[2] + u[1]))
    output[1] = x[1] + ( u[0] / u[1] * np.cos(x[2]) - u[0] / u[1] * np.cos(x[2] + u[1]))
    output[2] = x[2] + u[1] + u[2]
    return output


robot1_system = system()

robot1_system.process_model = process_model
robot1_system.process_covariance = np.eye(3) * 0.0001
robot1_system.process_input_size = 3
robot1_initial_state = np.zeros(3)



robot1_pf = particle_filter(robot1_system, num_particles= 100, initial_state= robot1_initial_state, initial_covariance=np.eye(3))

for odom_index in range(robot1_odometry.shape[0]):
    t = robot1_odometry[odom_index, 0]

    u = np.array([robot1_odometry[odom_index, 1], robot1_odometry[odom_index, 2], 0.0])

    robot1_pf.motion_step(u)
    robot1_pf.plot_particles(robot1_groundtruth[:,1].squeeze(), robot1_groundtruth[:,2].squeeze(), landmark_groundtruth[:,1].squeeze(), landmark_groundtruth[:,2].squeeze())


