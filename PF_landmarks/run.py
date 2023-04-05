import numpy as np
import os
import matplotlib.pyplot as plt

from PF import particle_filter, particle_filter_landmark, calculateMeanCovLandmarkPF
from utilities import plot, wrap2Pi




landmark_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Landmark_Groundtruth.dat'))
robot1_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Groundtruth.dat'))
robot1_measurement_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Measurement.dat'))
robot1_odometry_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Odometry.dat'))

landmark_groundtruth = np.loadtxt(landmark_groundtruth_path)
robot1_groundtruth = np.loadtxt(robot1_groundtruth_path)
robot1_measurements = np.loadtxt(robot1_measurement_path)
robot1_odometry = np.loadtxt(robot1_odometry_path)

class system:
    def __init__(self):
        self.process_model = None
        self.process_covariance = None
        self.process_input_size = None

        self.measurement_model = None
        self.measurement_covariance = None
        self.initial_state = None
        self.initial_covariance = None
        self.detected_landmarks = None

def process_model(x,u):

    output = np.zeros(3)
    output[0] = x[0] + (-u[0] / u[1] * np.sin(x[2]) + u[0] / u[1] * np.sin(x[2] + u[1]))
    output[1] = x[1] + ( u[0] / u[1] * np.cos(x[2]) - u[0] / u[1] * np.cos(x[2] + u[1]))
    output[2] = x[2] + u[1] + u[2]
    return output

def inv_measurement_model(x,z):
    output = np.zeros(2)
    output[0] = x[0] + z[0]* np.cos(x[2] + z[1])
    output[1] = x[1] + z[0] * np.sin(x[2] + z[1])
    return output

def measurement_model(x,landmark):
    output = np.zeros(2)
    output[0] = np.sqrt((landmark[1] - x[1])**2 + (landmark[0] - x[0])**2)
    output[1] = wrap2Pi(np.arctan2(landmark[1] - x[1], landmark[0] - x[0]) - x[2])
    return output


robot1_system = system()

robot1_system.process_model = process_model
robot1_system.process_covariance = np.eye(3) * 0.0001
robot1_system.process_input_size = 3
robot1_system.measurement_model = measurement_model
robot1_system.measurement_covariance = np.eye(2) * 0.0001
robot1_system.initial_state = np.zeros(3)
robot1_system.initial_covariance = np.eye(3) * 0.0001
robot1_system.detected_landmarks = {}




robot1_system.pf = particle_filter(robot1_system, num_particles= 100)






measurement_index = 0

plot(robot1_groundtruth[:,1].squeeze(), robot1_groundtruth[:,2].squeeze(), landmark_groundtruth[:,1].squeeze(), landmark_groundtruth[:,2].squeeze(), robot1_system)

for odom_index in range(robot1_odometry.shape[0]):
    t = robot1_odometry[odom_index, 0]

    u = np.array([robot1_odometry[odom_index, 1], robot1_odometry[odom_index, 2], 0.0])

    robot1_system.pf.motion_step(u)
    if robot1_measurements[measurement_index,0] == robot1_odometry[odom_index,0]:
        z = np.array([robot1_measurements[measurement_index,2], robot1_measurements[measurement_index,3]])

        # check if landmark ID has been detected before
        landmark_id = robot1_measurements[measurement_index,1]
        if landmark_id in robot1_system.detected_landmarks.keys():
            landmark_pf = robot1_system.detected_landmarks[landmark_id]
            likelihood_landmark_x = landmark_pf.update(z,robot1_system.pf)
            plot(robot1_groundtruth[:,1].squeeze(), robot1_groundtruth[:,2].squeeze(), landmark_groundtruth[:,1].squeeze(), landmark_groundtruth[:,2].squeeze(), robot1_system, additional_landmark = likelihood_landmark_x)
            landmark_pf.resampling()

            landmark_mean, landmark_cov = calculateMeanCovLandmarkPF(landmark_pf.particles.x)
            robot1_system.pf.measurement_step(z, landmark_mean, landmark_cov)


        else:
            landmark_pf = particle_filter_landmark(inv_measurement_model, robot1_system.measurement_covariance, z, robot1_system.pf, num_samples_per_robot_particle=10)
            robot1_system.detected_landmarks[landmark_id] = landmark_pf
        measurement_index += 1 


    plot(robot1_groundtruth[:,1].squeeze(), robot1_groundtruth[:,2].squeeze(), landmark_groundtruth[:,1].squeeze(), landmark_groundtruth[:,2].squeeze(), robot1_system)





