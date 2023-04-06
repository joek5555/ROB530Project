import numpy as np


def robot_system(self):

    def __init__(self):

        self.id = None
        self.process_model = None
        self.process_covariance = None
        self.process_input_size = None

        self.measurement_model = None
        self.measurement_covariance = None
        self.initial_state = None
        self.initial_covariance = None
        self.detected_landmarks = None

        self.odometry_index = None
        self.measurement_index = None