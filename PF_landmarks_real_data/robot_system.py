import numpy as np

class robot_system():

    def __init__(self):

        self.id = None
        self.process_model = None
        self.process_covariance = None
        self.process_input_size = None

        self.measurement_model = None
        self.measurement_covariance = None
        self.inverse_measurement_model = None

        self.initial_state = None
        self.initial_covariance = None
        self.detected_landmarks_pf = None

        self.odometry_data = None
        self.check_if_reached_end_of_odometry = None
        self.measurement_data = None
        self.odometry_index = None
        self.measurement_index = None
        self.check_if_reached_end_of_measurement = None

        self.robot_particle_color = None
        self.measurement_particle_color = None 

        self.pf = None

        self.alphas = None

        self.groundtruth_index = None

        self.means = None
        self.x_uncertainties = None
        self.y_uncertainties = None
        self.theta_uncertainties = None

    def get_means(self):
        return self.means


    def log_mean(self, timestamp):
        mean = self._get_mean()
        mean = np.insert(mean, 0, timestamp)
        added_mean = np.array([mean])
        self.means = np.append(self.means, added_mean, axis=0)

    
    def _get_mean(self):
        """
        Computes the mean pose of the robot based on the particles.
        
        Returns a numpy array of size 3 with [x, y, bearing]
        """
        particles = np.array(self.pf.particles.state)
        weights = np.array(self.pf.particles.weight)

        mean = np.average(particles, axis=0, weights=weights)
        mean = np.average(particles, axis=0)
        return mean
    
    def get_uncertainties(self):
        return self.x_uncertainties, self.y_uncertainties, self.theta_uncertainties
    
    def log_uncertainties(self, timestamp, cov_mat):
        x_uncertainty = cov_mat[0][0]
        y_uncertainty = cov_mat[1][1]
        theta_uncertainty = cov_mat[2][2]
        added_x_uncertainty = np.array([[timestamp, x_uncertainty]])
        added_y_uncertainty = np.array([[timestamp, y_uncertainty]])
        added_theta_uncertainty = np.array([[timestamp, theta_uncertainty]])
        self.x_uncertainties = np.append(self.x_uncertainties, added_x_uncertainty, axis=0)
        self.y_uncertainties = np.append(self.y_uncertainties, added_y_uncertainty, axis=0)
        self.theta_uncertainties = np.append(self.theta_uncertainties, added_theta_uncertainty, axis=0)