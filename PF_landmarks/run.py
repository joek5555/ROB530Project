import numpy as np

import yaml
from robot_system import robot_system
from models import process_model, measurement_model, inv_measurement_model
from utils import read_data, robot_sorting
from PF import particle_filter

with open("PF_landmarks/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)

print(param)

data = read_data()

robot_list = []

for i in range(param['num_robots']):
    robot = robot_system()
    robot.id = i+1
    robot.process_model = process_model
    robot.process_covariance = np.eye(3) * 0.0001
    robot.process_input_size = 3
    robot.measurement_model = measurement_model
    robot.measurement_covariance = np.eye(2) * 0.0001
    robot.initial_state = data.robots[i][0, 1:]
    robot.initial_covariance = np.eye(3) * 0.0001
    robot.detected_landmarks = {}

    robot.odometry_index = 0
    robot.measurement_index = 0

    robot.pf = particle_filter(robot, param['num_particles_robots'])

    robot_list.append(robot)


# main loop

current_runtime = 0.0

while current_runtime < param['max_runtime']:

    robot_list.sort(key=robot_sorting)

    robot = robot_list[0]

    if data.robots[robot.id -1].odometry[data_robot.odometry_index, 0], data_robot.measurement[data_robot.measurement_index, 0])
        
