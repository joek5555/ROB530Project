import numpy as np
from numpy.random import randn, rand
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class particles():
    def __init__(self):
        self.x = []
        self.weights = []

class particle_filter:

    def __init__(self, system, num_particles, initial_state, initial_covariance):

        self.process_model = system.process_model
        self.process_covariance = system.process_covariance
        self.process_covariance_L = np.linalg.cholesky(self.process_covariance)
        self.process_input_size = system.process_input_size

        self.measurement_model = system.measurement_model
        self.measurement_covariance = system.measurement_covariance
        #self.measurement_covariance_L = np.linalg.cholesky(self.measurement_covariance)

        self.num_particles = num_particles

        self.particles = particles()

        initial_covariance_L = np.linalg.cholesky(initial_covariance)

        for i in range(self.num_particles):
            #self.particles.x.append(np.dot(initial_covariance_L, randn(len(initial_state), 1)) + initial_state)
            self.particles.x.append(initial_state)
            self.particles.weights.append(1/self.num_particles)

    def motion_step(self, u):

        for i in range(self.num_particles):
            # generate noise to apply to the input
            sample_input_noise = self.process_covariance_L @ randn(self.process_input_size,1)
            print(sample_input_noise)

            self.particles.x[i] = self.process_model(self.particles.x[i], u + sample_input_noise.squeeze())
            #self.particles.x[i] = self.process_model(self.particles.x[i], u)


    def plot_particles(self,groundtruth_x, groundtruth_y, landmarks_x, landmarks_y):

        particles = np.array(self.particles.x)
        #print([particles])
        particles_x = particles[:,0]
        particles_y = particles[:,1]

        fig, ax = plt.subplots()
        ax.plot(groundtruth_x, groundtruth_y, 'r')
        ax.scatter(landmarks_x, landmarks_y, c='k', marker = "*")
        ax.scatter(particles_x, particles_y, s=1, c='b')


                
        ax.legend(['groundtruth','landmarks', 'particles'])

        plt.show()
