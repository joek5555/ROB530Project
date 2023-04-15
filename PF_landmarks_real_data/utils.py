import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class data_structure:
    def __init__(self):
        self.landmark_groundtruth = None
        self.robots = []

class robot_data_structure:
    def __init__(self):
        self.groundtruth = None
        self.measurements = None
        self.odometry = None

def read_data(max_runtime):
    data = data_structure()

    landmark_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Landmark_Groundtruth.dat'))
    robot1_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot1_Groundtruth.dat'))
    robot1_measurement_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot1_Measurement_x.dat'))
    robot1_odometry_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot1_Odometry.dat'))

    robot2_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot2_Groundtruth.dat'))
    robot2_measurement_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot2_Measurement_x.dat'))
    robot2_odometry_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot2_Odometry.dat'))

    data.landmark_groundtruth = np.loadtxt(landmark_groundtruth_path)

    max_runtime +=1248270000
    robot1 = robot_data_structure()
    robot1_groundtruth_long = np.loadtxt(robot1_groundtruth_path, dtype=np.longdouble)[:,0]
    robot1_measurements_long = np.loadtxt(robot1_measurement_path, dtype=np.longdouble)[:,0]
    robot1_odometry_long = np.loadtxt(robot1_odometry_path, dtype=np.longdouble)[:,0]
    robot1_groundtruth = np.loadtxt(robot1_groundtruth_path)[:,1:]
    robot1_measurements = np.loadtxt(robot1_measurement_path)[:,1:]
    robot1_odometry = np.loadtxt(robot1_odometry_path)[:,1:]

    for i in range(robot1_groundtruth_long.shape[0]):
        if robot1_groundtruth_long[i] > max_runtime:
            robot1_groundtruth_long = robot1_groundtruth_long[0:i+1]
            robot1_groundtruth = robot1_groundtruth[0:i+1, :]
            break
    for i in range(robot1_measurements.shape[0]):
        if robot1_measurements_long[i] > max_runtime:
            robot1_measurements_long = robot1_measurements_long[0:i]
            robot1_measurements = robot1_measurements[0:i, :]
            break
    for i in range(robot1_odometry.shape[0]):
        if robot1_odometry_long[i] > max_runtime:
            robot1_odometry_long = robot1_odometry_long[0:i]
            robot1_odometry = robot1_odometry[0:i, :]
            break

    robot1.groundtruth = np.hstack([(robot1_groundtruth_long - 1248270000).reshape(-1,1), robot1_groundtruth]).astype(np.float64)
    robot1.measurements = np.hstack([(robot1_measurements_long - 1248270000).reshape(-1,1), robot1_measurements]).astype(np.float64) 
    robot1.odometry = np.hstack([(robot1_odometry_long - 1248270000).reshape(-1,1), robot1_odometry]).astype(np.float64) 

    robot2 = robot_data_structure()
    robot2_groundtruth_long = np.loadtxt(robot2_groundtruth_path, dtype=np.longdouble)[:,0]
    robot2_measurements_long = np.loadtxt(robot2_measurement_path, dtype=np.longdouble)[:,0]
    robot2_odometry_long = np.loadtxt(robot2_odometry_path, dtype=np.longdouble)[:,0]
    robot2_groundtruth = np.loadtxt(robot2_groundtruth_path)[:,1:]
    robot2_measurements = np.loadtxt(robot2_measurement_path)[:,1:]
    robot2_odometry = np.loadtxt(robot2_odometry_path)[:,1:]

    for i in range(robot2_groundtruth_long.shape[0]):
        if robot2_groundtruth_long[i] > max_runtime:
            robot2_groundtruth_long = robot2_groundtruth_long[0:i+1]
            robot2_groundtruth = robot2_groundtruth[0:i+1, :]
            break
    for i in range(robot2_measurements.shape[0]):
        if robot2_measurements_long[i] > max_runtime:
            robot2_measurements_long = robot2_measurements_long[0:i]
            robot2_measurements = robot2_measurements[0:i, :]
            break
    for i in range(robot2_odometry.shape[0]):
        if robot2_odometry_long[i] > max_runtime:
            robot2_odometry_long = robot2_odometry_long[0:i]
            robot2_odometry = robot2_odometry[0:i, :]
            break

    robot2.groundtruth = np.hstack([(robot2_groundtruth_long - 1248270000).reshape(-1,1), robot2_groundtruth]).astype(np.float64)
    robot2.measurements = np.hstack([(robot2_measurements_long - 1248270000).reshape(-1,1), robot2_measurements]).astype(np.float64)
    robot2.odometry = np.hstack([(robot2_odometry_long - 1248270000).reshape(-1,1), robot2_odometry]).astype(np.float64) 


    

    data.robots = [robot1, robot2]


    return(data)

def calculateMeanCovFromList(particle_list):
    """
    particle_array = np.array(particle_list)
    particle_mean = np.sum(particle_array, axis=0) / particle_array.shape[0]

    zero_mean = particle_array - np.tile(particle_mean, (particle_array.shape[0], 1))
    

    particle_covariance = zero_mean.T @ zero_mean / particle_array.shape[0]

    # check to raise error if not positive definite
    np.linalg.cholesky(particle_covariance)
    print("ran calculateMeanCovFromList correctly")


    return particle_mean, particle_covariance
    """
    """
    particle_array = np.array(particle_list)

    #particle_mean = np.sum(particle_array, axis=0) / particle_array.shape[0]
    particle_mean = np.mean(particle_array, axis = 0)
    print(particle_array)
    print(particle_mean)
    print(particle_mean.shape[0])

    if particle_mean.shape[0] == 3:
        sinSum = 0
        cosSum = 0
        for s in range(particle_array.shape[0]):
            cosSum += np.cos(particle_array[s,2])
            sinSum += np.sin(particle_array[s,2])
        particle_mean[2] = np.arctan2(sinSum, cosSum)

    zero_mean = np.zeros_like(particle_array) 
    for s in range(particle_array.shape[0]):
        zero_mean[s,:] = particle_array[s,:] - particle_mean
        if particle_mean.shape[0] == 3:
            zero_mean[s,2] = wrap2Pi(zero_mean[s,2]) 

    particle_covariance = zero_mean.T @ zero_mean / particle_array.shape[0]
    np.linalg.cholesky(particle_covariance)
    print("ran calculateMeanCovFromList correctly")
    print(particle_covariance)

    return particle_mean, particle_covariance
    """
    # to ensure mean covariance are correct, transpose array and then use code from HW5
    particle_array = np.array(particle_list)
    particle_array = particle_array.T
    particle_mean = np.mean(particle_array, axis=1)
    if particle_mean.shape[0] == 3: # only mess with heading if this is a robot pose
        sinSum = 0
        cosSum = 0
        for s in range(particle_array.shape[1]):
            cosSum += np.cos(particle_array[2,s])
            sinSum += np.sin(particle_array[2,s])
        particle_mean[2] = np.arctan2(sinSum, cosSum)
    zero_mean = np.zeros_like(particle_array)
    for s in range(particle_array.shape[1]):
        zero_mean[:,s] = particle_array[:,s] - particle_mean
        if particle_mean.shape[0] == 3: # only mess with heading if this is a robot pose
            zero_mean[2,s] = wrap2Pi(zero_mean[2,s])
    particle_covariance = zero_mean @ zero_mean.T / particle_array.shape[1]
    
    return particle_mean, particle_covariance


def wrap2Pi(input):
    phases =  (( -input + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0

    return phases


def robot_sorting(robot):

    return min(max(robot.odometry_data[robot.odometry_index, 0], robot.check_if_reached_end_of_odometry), 
               max(robot.measurement_data[robot.measurement_index, 0], robot.check_if_reached_end_of_measurement))

def getLandmarkParticles(z, inv_measurement_model, measurement_covariance, robot_particles, num_particles_per_robot_particle = 10):
    
    list_landmark_particles = []
    measurement_covariance_L = np.linalg.cholesky(measurement_covariance)

    for particle in robot_particles:
            detected_landmark_location = inv_measurement_model(particle.squeeze(), z)

            for i in range (num_particles_per_robot_particle):
                sample_measurement_noise = measurement_covariance_L @ np.random.randn(z.shape[0],1)
                list_landmark_particles.append(detected_landmark_location + sample_measurement_noise.squeeze())
    
    return list_landmark_particles

"""    
def plot_covariance(ax, mean, cov):
    xy = (mean[0], mean[1])
    a = cov[0, 0]
    b = cov[0, 1]
    c = cov[1, 1]

    term_1 = (a + c) / 2
    term_2 = np.sqrt((((a - c) / 2) ** 2) + b ** 2)

    lambda_1 = term_1 + term_2
    lambda_2 = term_1 - term_2

    if b == 0:
        if a >= c:
            theta = 0
        else:
            theta = 45
    else:
        theta = np.arctan2(lambda_1 - a, b)

    ellipse = Ellipse(xy, width=2 * lambda_1, height=2 * lambda_2, angle=theta, fill=False, color='red')
    ax.add_patch(ellipse)
"""
def plot_covariance(ax, mean, covariance):
    covariance = covariance[0:2, 0:2]
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    if eigenvalues[0] > eigenvalues[1]:
        largest_eigenvalue = eigenvalues[0]
        smallest_eigenvalue= eigenvalues[1]
        largest_eigenvector = eigenvectors[:,0]
        smallest_eigenvector = eigenvectors[:,1]
    else:
        largest_eigenvalue = eigenvalues[1]
        smallest_eigenvalue = eigenvalues[0]
        largest_eigenvector = eigenvectors[:,1].reshape(-1)
        smallest_eigenvector = eigenvectors[:,0].reshape(-1)
    # calulate angle between largest eigenvector and x_axis
    angle = np.arctan2(largest_eigenvector[1], largest_eigenvector[0])
    if angle < 0:
        angle = angle + 2*np.pi

    # get the 95% confidence interval error ellipse
    chisquare_val = 2.4477
    theta_grid = np.linspace(0,2*np.pi, 100)
    phi = angle
    a = chisquare_val * np.sqrt(largest_eigenvalue)
    b = chisquare_val * np.sqrt(smallest_eigenvalue)
    # elipse in x and y coordinates
    ellipse_x_r  = a*np.cos( theta_grid )
    ellipse_y_r  = b*np.sin( theta_grid )

    #Define a rotation matrix
    R = np.array([[np.cos(phi), np.sin(phi)],[ -np.sin(phi), np.cos(phi)]])

    #let's rotate the ellipse to some angle phi
    r_ellipse = np.vstack((ellipse_x_r,ellipse_y_r)).T @ R
    ax.plot(r_ellipse[:,0] + mean[0], r_ellipse[:,1] + mean[1], c='k')


def plot(robots, data, image_num, current_time, label, observed_landmark_particles=None, robot_observing = None, groundtruth_point = None):

    path_to_images = os.path.realpath(os.path.join(os.path.dirname(__file__), 'saved_images'))
    if os.path.exists(path_to_images) == False:
       os.mkdir(path_to_images) 

    fig, axs = plt.subplots(len(robots))
    fig.tight_layout(h_pad=2)

    robot_groundtruth_colors = ['r', 'y', 'g', 'b', 'm']

    for i, robot in enumerate(robots):

        axs[i].set_title("R" + str(i+1) + ": dark_color = robot, light_color= landmark, orange = observed")
        
        landmarks_x = data.landmark_groundtruth[:, 1]
        landmarks_y = data.landmark_groundtruth[:,2]
        axs[i].scatter(landmarks_x, landmarks_y, c='k', marker = "*")

        for j, robot_groundtruth_data in enumerate(data.robots):
            robot_groundtruth_x = robot_groundtruth_data.groundtruth[:,1]
            robot_groundtruth_y = robot_groundtruth_data.groundtruth[:,2]
            axs[i].plot(robot_groundtruth_x, robot_groundtruth_y, robot_groundtruth_colors[j])

            robot_particles_x = (np.array(robot.pf.particles.state))[:,0]
            robot_particles_y = (np.array(robot.pf.particles.state))[:,1]
            axs[i].scatter(robot_particles_x, robot_particles_y, s=1, c= robot.robot_particle_color)
            robot_mean, robot_cov = calculateMeanCovFromList(robot.pf.particles.state)
            plot_covariance(axs[i], robot_mean, robot_cov)

        for landmark_id, landmark_pf in robot.detected_landmarks_pf.items():
            landmark_particles_x = (np.array(landmark_pf.particles.state))[:,0]
            landmark_particles_y = (np.array(landmark_pf.particles.state))[:,1]
            axs[i].scatter(landmark_particles_x, landmark_particles_y, s=1, c=robot.measurement_particle_color)
            landmark_mean, landmark_cov = calculateMeanCovFromList(landmark_pf.particles.state)
            plot_covariance(axs[i], landmark_mean, landmark_cov)

    if groundtruth_point is not None:
        axs[robot_observing - 1].scatter(groundtruth_point[1], groundtruth_point[2],s = 200, c='#6b6666', marker = "x")

    if observed_landmark_particles is None:
        axs[0].set_xlim([2.5,3.8])
        axs[0].set_ylim([-3.5,-2.2])
        axs[1].set_xlim([0.5,1.5])
        axs[1].set_ylim([-1.5,-0.5])
        plt.savefig(path_to_images + "/image_" + str(image_num) + "_"  + label +"_time_" + str(round(current_time, 2)) + ".png")
        plt.close()

    else:

        observed_landmark_particles_x = (np.array(observed_landmark_particles))[:,0]
        observed_landmark_particles_y = (np.array(observed_landmark_particles))[:,1]

        axs[robot_observing - 1].scatter(observed_landmark_particles_x, observed_landmark_particles_y, s=1, c='#ed8026', marker = "x")
        observed_landmark_mean, observed_landmark_cov = calculateMeanCovFromList(observed_landmark_particles)
        plot_covariance(axs[robot_observing - 1], observed_landmark_mean, observed_landmark_cov)

        plt.savefig(path_to_images + "/image_" + str(image_num) + "_" + label +"_time_" + str(round(current_time,2)) + ".png")
        plt.close()
        
