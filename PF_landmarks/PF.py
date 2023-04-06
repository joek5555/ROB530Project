import numpy as np
from numpy.random import randn, rand
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from utils import wrap2Pi

class particles():
    def __init__(self):
        self.x = []
        self.weights = []

class particle_filter:

    def __init__(self, system, num_particles):

        self.process_model = system.process_model
        self.process_covariance = system.process_covariance
        self.process_covariance_L = np.linalg.cholesky(self.process_covariance)
        self.process_input_size = system.process_input_size

        self.measurement_model = system.measurement_model
        self.measurement_covariance = system.measurement_covariance
        #self.measurement_covariance_L = np.linalg.cholesky(self.measurement_covariance)
        self.initial_state = system.initial_state


        self.initial_covariance = system.initial_covariance

        self.num_particles = num_particles

        self.particles = particles()

        initial_covariance_L = np.linalg.cholesky(self.initial_covariance)

        for i in range(self.num_particles):

            self.particles.x.append((np.dot(initial_covariance_L, randn(self.initial_state.shape[0], 1)) + self.initial_state).reshape(-1))
            self.particles.weights.append(1/self.num_particles)

    def motion_step(self, u):

        for i in range(self.num_particles):
            # generate noise to apply to the input
            sample_input_noise = self.process_covariance_L @ randn(self.process_input_size,1)

            self.particles.x[i] = self.process_model(self.particles.x[i], u + sample_input_noise.squeeze())
            #self.particles.x[i] = self.process_model(self.particles.x[i], u)

    def measurement_step(self, z, landmark_mean, landmark_cov):
        
        new_weights = np.zeros([self.num_particles])
        for i in range(self.num_particles):
            z_pred = self.measurement_model(self.particles.x[i], landmark_mean)

            innovation = z - z_pred
            innovation[1] = wrap2Pi(innovation[1])

            new_weights[i] = multivariate_normal.pdf(innovation, np.array([0,0]), landmark_cov + self.measurement_covariance)

        # update weights using Bayes Rule
        new_weights = np.multiply(np.array(self.particles.weights), new_weights)
        # normalize
        new_weights = new_weights/ np.sum(new_weights)

        self.particles.weights = new_weights.tolist()

        self.resampling()




    def resampling(self):
        # low variance resampling

        particles_x = np.array(self.particles.x)
        particles_weights = np.array(self.particles.weights)
        self.particles.x.clear()
        self.particles.weights.clear()

        W = np.cumsum(particles_weights)

        r = np.random.rand(1) / self.num_particles
        # r = 0.5 / self.n
        j = 0 # one change, j = 1 in original, but then you can never select first particle, even if it has high weight
        for i in range(self.num_particles):

            u = r +  i/ self.num_particles
            while u > W[j]:
                j = j + 1
            self.particles.x.append(particles_x[j, :])
            self.particles.weights.append( 1 / self.num_particles)