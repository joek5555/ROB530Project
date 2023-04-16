import numpy as np

import yaml
from robot_system import robot_system
import models
from utils import read_data, robot_sorting, calculateMeanCovFromList, getLandmarkParticles, plot, plot_robot_paths_and_error
from PF import particle_filter

# settings.yaml contains many of the parameters to be tuned
# this opens up settings.yaml and saves the data in the param object
with open("PF_landmarks_real_data/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)

# read_data() will read information from the .dat files and save them into numpy arrays
data = read_data(param['max_runtime'])


robot_list = []

# create a robot system class for the number of robots you are working with
# and add them to the robot list
for i in range(param['num_robots']):
    robot = robot_system()
    robot.id = i+1
    robot.process_model = models.process_model
    robot.process_covariance = models.process_model_noise
    robot.process_input_size = 3

    robot.measurement_model = models.measurement_model
    robot.measurement_covariance = np.array([[param['robot_measurement_range_covariance'],0],[0, param['robot_measurement_bearing_covariance']]])
    robot.inverse_measurement_model = models.inv_measurement_model

    robot.initial_state = data.robots[i].groundtruth[0, 1:]
    robot.initial_covariance = np.eye(3) * param['robot_initial_covariance']
    robot.detected_landmarks_pf = {}

    robot.odometry_data = data.robots[i].odometry
    robot.measurement_data = data.robots[i].measurements
    robot.odometry_index = 0
    robot.check_if_reached_end_of_odometry = 0
    robot.measurement_index = 0
    robot.check_if_reached_end_of_measurement = 0

    timestamped_mean = robot.initial_state
    timestamped_mean = np.insert(timestamped_mean, 0, data.robots[i].groundtruth[0, 0])
    robot.means = np.array([timestamped_mean])

    robot.robot_particle_color = param['robot_particle_color'][i]
    robot.measurement_particle_color = param['measurement_particle_color'][i]

    robot.pf = particle_filter(
        process_model = robot.process_model, process_covariance = robot.process_covariance, process_input_size=robot.process_input_size,
        measurement_model= robot.measurement_model, measurement_covariance= robot.measurement_covariance,
        initial_state= robot.initial_state, initial_covariance= robot.initial_covariance,
        num_particles = param['num_particles_robots'])

    robot.alphas = np.array(param['robot_alphas_sqrt']) * np.array(param['robot_alphas_sqrt'])

    robot.groundtruth_index = 0

    robot_list.append(robot)


# main loop where we either perform a motion_step or measurement step based on 
# the odometry or measurement data that occurs next 

image_num = 0
# decide if you want to plot the measurement and motion step for each robot
plot_motion_step = [True, True]
plot_measurement_step = [True, True]

while True:
    print(image_num)

    # sort the robot list by the sorting robot function
    # this sorts the robots in acending timestep, where the timestep is the 
    # minimum value between the next robot odometry timestep and the 
    # next robot measurement timestep
    # the first robot in the sorted list will have the next move or update step
    robot = sorted(robot_list, key=robot_sorting)[0]


    # if the next motion or measurement step would use data from a timestep that is after max_runtime
    # or if we have reached the end of both datasets, break from while loop
    if (max(robot.odometry_data[robot.odometry_index, 0], robot.check_if_reached_end_of_odometry) >= param['max_runtime'] and 
        max(robot.measurement_data[robot.measurement_index, 0], robot.check_if_reached_end_of_measurement) >= param['max_runtime']):
        break

    # check to see if the robot should take a move step or measurement step
    if (max(robot.odometry_data[robot.odometry_index, 0], robot.check_if_reached_end_of_odometry) < 
        max(robot.measurement_data[robot.measurement_index, 0], robot.check_if_reached_end_of_measurement)):
        
        # odometry timestep is sooner

        indices = data.robots[robot.id-1].groundtruth[robot.groundtruth_index:,0] > robot.odometry_data[robot.odometry_index, 0]
        all_greater = data.robots[robot.id-1].groundtruth[robot.groundtruth_index:,:]
        all_greater = all_greater[indices, :]
        if len(all_greater) == 0:
            break
        groundtruth = all_greater[0,:]     
        robot.groundtruth_index += 1 + np.where(indices)[0][0]

        timestamp = robot.odometry_data[robot.odometry_index, 0]

        # calculate input u at this timestep
        u = np.array([robot.odometry_data[robot.odometry_index, 1], robot.odometry_data[robot.odometry_index, 2], 0.0])
        if(u[0] != 0 or u[1] != 0):
            # perform the motion step
            if robot.odometry_index == 0:
                dt = robot.odometry_data[robot.odometry_index+1, 0] - robot.odometry_data[robot.odometry_index, 0]
            else:
                dt = robot.odometry_data[robot.odometry_index, 0] - robot.odometry_data[robot.odometry_index - 1, 0]
            
            robot.pf.motion_step(u, dt, robot.alphas)
            
            # if the robot did not have a forward velocity, then we will not be able to visually see the result
            # so do not plot
            
            if(u[0] != 0 and plot_motion_step[robot.id-1]):
                label = "motion"
                plot(robot_list, data, image_num, robot.odometry_data[robot.odometry_index, 0], label, robot_observing=robot.id, groundtruth_point = groundtruth)
                image_num += 1
            

        robot.odometry_index += 1
        # if index is past the end of the data, set the check_if_reached_end_of_odometry
        if robot.odometry_index >= robot.odometry_data.shape[0]:
            robot.check_if_reached_end_of_odometry = param['max_runtime']
            robot.odometry_index = 0 # no longer use odometry index, ensures it is not out of scope of data


        

    else:
         
        # measurement timestep is sooner

        # assume that since odometry comes in faster than measurements, the groundtruth for the last odometry 
        # time is close enough to this measurement time
        groundtruth = data.robots[robot.id-1].groundtruth[robot.groundtruth_index-1,:]

        # get measurement z at current timestep
        z = np.array([robot.measurement_data[robot.measurement_index, 2], robot.measurement_data[robot.measurement_index, 3]])
        landmark_id = int(robot.measurement_data[robot.measurement_index, 1])
        timestamp = robot.measurement_data[robot.measurement_index, 0]

        # if the landmark id is less than or equal to the number of robots, you have detected a robot
        if landmark_id <= param['num_robots'] or landmark_id == 3 or landmark_id == 4 or landmark_id == 5:
            pass
            """
            # if you detect another robot
            # 'communicate' with the robot and get the mean and covariance associated with its location
            detected_robot = robot_list[landmark_id-1]
            detected_robot_mean, detected_robot_covariance = calculateMeanCovFromList(detected_robot.pf.particles.state)
            # then use this mean and variance as a measurement update
            robot.pf.measurement_step(z, detected_robot_mean[0:2], detected_robot_covariance[0:2, 0:2])

            for landmark_id, landmark_pf in robot.detected_landmarks_pf.items():
                 if landmark_id in detected_robot.detected_landmarks_pf.keys():
                    detected_robot_landmark_particles = detected_robot.detected_landmarks_pf[landmark_id].particles.state
                    detected_robot_landmark_mean, detected_robot_landmark_covariance = calculateMeanCovFromList(detected_robot_landmark_particles)  
                    robot.detected_landmarks_pf[landmark_id].measurement_step_landmarks(detected_robot_landmark_mean, detected_robot_landmark_covariance)
            """
        
        elif landmark_id in robot.detected_landmarks_pf.keys():

            # if you detect a landmark you have detected before
            # get a list of particles representing where this landmark could be based on where the robot
            # currently thinks it is and the inverse_measurement_model
            detected_landmark_particles = getLandmarkParticles(z, robot.inverse_measurement_model, robot.measurement_covariance, 
                                                               robot.pf.particles.state, param['num_measurement_particles_per_robot_particle'])
            # calculate the mean and covariance of this detected landmark, then use that as a measurement update for the landmark pf
            current_landmark_mean, current_landmark_covariance = calculateMeanCovFromList(robot.detected_landmarks_pf[landmark_id].particles.state)
            detected_landmark_mean, detected_landmark_covariance = calculateMeanCovFromList(detected_landmark_particles)
            
            #robot.detected_landmarks_pf[landmark_id].measurement_step_compare_particles(detected_landmark_particles)

            # plot the detected landmark so that we can see if measurement update was reasonable
            if plot_measurement_step[robot.id-1]:
                label = "R" + str(robot.id) + "_land_update"
                plot(robot_list, data, image_num, robot.measurement_data[robot.measurement_index, 0], label,
                    observed_landmark_particles = detected_landmark_particles, robot_observing = robot.id, groundtruth_point = groundtruth)
                
            #update the landmark's particle filter
            robot.detected_landmarks_pf[landmark_id].measurement_step_landmarks(detected_landmark_mean, detected_landmark_covariance)
            #robot.detected_landmarks_pf[landmark_id].measurement_step_compare_particles(detected_landmark_particles)
            #robot.detected_landmarks_pf[landmark_id].measurement_step_combine_gaussians(detected_landmark_mean, detected_landmark_covariance)
            
            if plot_measurement_step[robot.id-1]:
                label = "R" + str(robot.id) + "_land_updated"
                plot(robot_list, data, image_num, robot.measurement_data[robot.measurement_index, 0], label,
                    observed_landmark_particles = [np.array([0,0])], robot_observing = robot.id, groundtruth_point = groundtruth)
                
            # now that the landmark has been detected at least twice, we can update our robot position based on this landmark measurement
            landmark_mean, landmark_covariance = calculateMeanCovFromList(robot.detected_landmarks_pf[landmark_id].particles.state)
            robot.pf.measurement_step(z, landmark_mean, landmark_covariance)


        else:
             
            # if this is the first time you have detected the landmark
            # get a list of particles representing where this landmark could be based on where the robot
            # currently thinks it is and the inverse_measurement_model
            detected_landmark_particles = getLandmarkParticles(z, robot.inverse_measurement_model, robot.measurement_covariance, 
                                                               robot.pf.particles.state, param['num_measurement_particles_per_robot_particle'])
            # create a particle filter associated with this landmark and add it to the robot.detected_landmarks_pf list
            robot.detected_landmarks_pf[landmark_id] = particle_filter( 
                given_starting_particles = detected_landmark_particles)

        
        # plot the measurement step
        if plot_measurement_step[robot.id-1]:
            label = "R" + str(robot.id) + "measure"
            plot(robot_list, data, image_num, robot.measurement_data[robot.measurement_index, 0], label, robot_observing=robot.id, groundtruth_point = groundtruth)
            image_num += 1
        
        robot.measurement_index += 1
        # if index is past the end of the data, set the check_if_reached_end_of_measurement
        if robot.measurement_index >= robot.measurement_data.shape[0]:
            robot.check_if_reached_end_of_measurement = param['max_runtime']
            robot.measurement_index = 0 # no longer use measurement index, ensures it is not out of scope of data

        
    # Keep track of the estimate
    robot.log_mean(timestamp)
    

plot_robot_paths_and_error(data, robot_list)
