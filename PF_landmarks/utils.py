import numpy as np
import os
import matplotlib.pyplot as plt


class data_structure:
    pass

def read_data():
    data = data_structure()

    landmark_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Landmark_Groundtruth.dat'))
    robot1_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Groundtruth.dat'))
    robot1_measurement_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Measurement.dat'))
    robot1_odometry_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot1_Odometry.dat'))

    robot2_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot2_Groundtruth.dat'))
    robot2_measurement_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot2_Measurement.dat'))
    robot2_odometry_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'PF_landmarks/test_data', 'Robot2_Odometry.dat'))

    data.landmark_groundtruth = np.loadtxt(landmark_groundtruth_path)

    robot1 = data_structure()
    robot1.groundtruth = np.loadtxt(robot1_groundtruth_path)
    robot1.measurements = np.loadtxt(robot1_measurement_path)
    robot1.odometry = np.loadtxt(robot1_odometry_path)

    robot2 = data_structure()
    robot2.groundtruth = np.loadtxt(robot2_groundtruth_path)
    robot2.measurements = np.loadtxt(robot2_measurement_path)
    robot2.odometry = np.loadtxt(robot2_odometry_path)

    data.robots = [robot1, robot2]

    return(data)




def wrap2Pi(input):
    phases =  (( -input + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0

    return phases


def robot_sorting(data_robot):
    return min(data_robot.odometry[data_robot.odometry_index, 0], data_robot.measurement[data_robot.measurement_index, 0])