import numpy as np
import os
import matplotlib.pyplot as plt

from PF import particle_filter, particle_filter_landmark, calculateMeanCovLandmarkPF
from utilities import plot, wrap2Pi




landmark_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Landmark_Groundtruth.dat'))
robot1_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Groundtruth.dat'))
robot1_measurement_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Measurement.dat'))
robot1_odometry_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Odometry.dat'))

robot2_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot2_Groundtruth.dat'))
robot2_measurement_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot2_Measurement.dat'))
robot2_odometry_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot2_Odometry.dat'))

landmark_groundtruth = np.loadtxt(landmark_groundtruth_path)
robot1_groundtruth = np.loadtxt(robot1_groundtruth_path)
robot1_measurements = np.loadtxt(robot1_measurement_path)
robot1_odometry = np.loadtxt(robot1_odometry_path)

robot2_groundtruth = np.loadtxt(robot2_groundtruth_path)
robot2_measurements = np.loadtxt(robot2_measurement_path)
robot2_odometry = np.loadtxt(robot2_odometry_path)

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
robot1_system.initial_state = robot1_groundtruth[0, 1:]
robot1_system.initial_covariance = np.eye(3) * 0.0001
robot1_system.detected_landmarks = {}

robot1_system.pf = particle_filter(robot1_system, num_particles= 100)

robot2_system = system()

robot2_system.process_model = process_model
robot2_system.process_covariance = np.eye(3) * 0.0001
robot2_system.process_input_size = 3
robot2_system.measurement_model = measurement_model
robot2_system.measurement_covariance = np.eye(2) * 0.0001
robot2_system.initial_state = robot2_groundtruth[0, 1:]
robot2_system.initial_covariance = np.eye(3) * 0.0001
robot2_system.detected_landmarks = {}

robot2_system.pf = particle_filter(robot2_system, num_particles= 100)



num_robots = 2
percent_to_resample_landmarks = 1
robot1_measurement_index = 0
robot2_measurement_index = 0

plot(robot1_groundtruth[:,1].squeeze(), robot1_groundtruth[:,2].squeeze(), robot2_groundtruth[:,1].squeeze(), robot2_groundtruth[:,2].squeeze(), 
     landmark_groundtruth[:,1].squeeze(), landmark_groundtruth[:,2].squeeze(), robot1_system, robot2_system)

for odom_index in range(robot1_odometry.shape[0]):
    t = robot1_odometry[odom_index, 0]

    u = np.array([robot1_odometry[odom_index, 1], robot1_odometry[odom_index, 2], 0.0])

    robot1_system.pf.motion_step(u)
    if robot1_measurements[robot1_measurement_index,0] == robot1_odometry[odom_index,0]:
        z = np.array([robot1_measurements[robot1_measurement_index,2], robot1_measurements[robot1_measurement_index,3]])

        # check if landmark ID has been detected before
        landmark_id = robot1_measurements[robot1_measurement_index,1]

        if landmark_id <= num_robots:
            detected_robot_mean, detected_robot_covariance = calculateMeanCovLandmarkPF(robot1_system.pf.particles.x)
            robot1_system.pf.measurement_step(z, detected_robot_mean[0:2], detected_robot_covariance[0:2, 0:2])

        elif landmark_id in robot1_system.detected_landmarks.keys():
            landmark_pf = robot1_system.detected_landmarks[landmark_id]
            likelihood_landmark_x = landmark_pf.update(z,robot1_system.pf)
            plot(robot1_groundtruth[:,1].squeeze(), robot1_groundtruth[:,2].squeeze(), robot2_groundtruth[:,1].squeeze(), robot2_groundtruth[:,2].squeeze(), 
                 landmark_groundtruth[:,1].squeeze(), landmark_groundtruth[:,2].squeeze(), robot1_system, robot2_system, 
                 additional_landmark = likelihood_landmark_x, main_robot=1, timestep = odom_index)

            #landmark_pf.resampling()
            landmark_pf.resampling_sample_normal(int(landmark_pf.num_particles * percent_to_resample_landmarks))

            landmark_mean, landmark_cov = calculateMeanCovLandmarkPF(landmark_pf.particles.x)
            robot1_system.pf.measurement_step(z, landmark_mean, landmark_cov)
            print("landmark 1 update")
            print(landmark_mean)
            print(landmark_cov)


        else:
            landmark_pf = particle_filter_landmark(inv_measurement_model, robot1_system.measurement_covariance, z, robot1_system.pf, num_samples_per_robot_particle=10)
            robot1_system.detected_landmarks[landmark_id] = landmark_pf
            landmark_mean, landmark_cov = calculateMeanCovLandmarkPF(landmark_pf.particles.x)
            print("landmark 1 initialize")
            print(landmark_mean)
            print(landmark_cov)
            
        robot1_measurement_index += 1 


    plot(robot1_groundtruth[:,1].squeeze(), robot1_groundtruth[:,2].squeeze(), robot2_groundtruth[:,1].squeeze(), robot2_groundtruth[:,2].squeeze(), 
     landmark_groundtruth[:,1].squeeze(), landmark_groundtruth[:,2].squeeze(), robot1_system, robot2_system, main_robot=1, timestep = odom_index)


    # robot 2
    t = robot1_odometry[odom_index, 0]

    u = np.array([robot2_odometry[odom_index, 1], robot2_odometry[odom_index, 2], 0.0])

    robot2_system.pf.motion_step(u)

    if robot2_measurements[robot2_measurement_index,0] == robot1_odometry[odom_index,0]:
        z = np.array([robot2_measurements[robot2_measurement_index,2], robot2_measurements[robot2_measurement_index,3]])

        # check if landmark ID has been detected before
        landmark_id = robot2_measurements[robot2_measurement_index,1]

        if landmark_id <= num_robots:
            detected_robot_mean, detected_robot_covariance = calculateMeanCovLandmarkPF(robot1_system.pf.particles.x)
            robot2_system.pf.measurement_step(z, detected_robot_mean[0:2], detected_robot_covariance[0:2, 0:2])

        elif landmark_id in robot2_system.detected_landmarks.keys():
            landmark_pf = robot2_system.detected_landmarks[landmark_id]
            likelihood_landmark_x = landmark_pf.update(z,robot2_system.pf)
            plot(robot1_groundtruth[:,1].squeeze(), robot1_groundtruth[:,2].squeeze(), robot2_groundtruth[:,1].squeeze(), robot2_groundtruth[:,2].squeeze(), 
                 landmark_groundtruth[:,1].squeeze(), landmark_groundtruth[:,2].squeeze(), robot1_system, robot2_system, 
                 additional_landmark = likelihood_landmark_x, main_robot=2, timestep = odom_index)

            #landmark_pf.resampling()
            landmark_pf.resampling_sample_normal(int(landmark_pf.num_particles * percent_to_resample_landmarks))

            landmark_mean, landmark_cov = calculateMeanCovLandmarkPF(landmark_pf.particles.x)
            robot2_system.pf.measurement_step(z, landmark_mean, landmark_cov)

            print("landmark 2 update")
            print(landmark_mean)
            print(landmark_cov)
            

        else:
            landmark_pf = particle_filter_landmark(inv_measurement_model, robot2_system.measurement_covariance, z, robot2_system.pf, num_samples_per_robot_particle=10)
            robot2_system.detected_landmarks[landmark_id] = landmark_pf

            landmark_mean, landmark_cov = calculateMeanCovLandmarkPF(landmark_pf.particles.x)

            print("landmark 2 initialize")
            print(landmark_mean)
            print(landmark_cov)
        
        robot2_measurement_index += 1 


    plot(robot1_groundtruth[:,1].squeeze(), robot1_groundtruth[:,2].squeeze(), robot2_groundtruth[:,1].squeeze(), robot2_groundtruth[:,2].squeeze(), 
     landmark_groundtruth[:,1].squeeze(), landmark_groundtruth[:,2].squeeze(), robot1_system, robot2_system, main_robot=2, timestep = odom_index)


