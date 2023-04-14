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