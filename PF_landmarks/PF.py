import numpy as np
from numpy.random import randn, rand
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from utilities import wrap2Pi

class particles():
    def __init__(self, num_particles, dimensions):
        self.x = np.zeros([num_particles, dimensions])
        self.weights = np.zeros([num_particles])

class particle_filter:

    def __init__(self, system, num_particles):

        self.dimensions = system.dimensions
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

        self.particles = particles(self.num_particles, self.dimensions)

        initial_covariance_L = np.linalg.cholesky(self.initial_covariance)

        for i in range(self.num_particles):
            #self.particles.x.append(np.dot(initial_covariance_L, randn(len(initial_state), 1)) + initial_state)
            self.particles.x[i, :] = self.initial_state
            self.particles.weights[i] = 1/self.num_particles

    def motion_step(self, u):

        for i in range(self.num_particles):
            # generate noise to apply to the input
            sample_input_noise = self.process_covariance_L @ randn(self.process_input_size,1)

            self.particles.x[i,:] = self.process_model(self.particles.x[i,:].squeeze(), u + sample_input_noise.squeeze())


    def measurement_step(self, z, landmark_mean, landmark_cov):
        
        new_weights = np.zeros([self.num_particles])
        for i in range(self.num_particles):
            z_pred = self.measurement_model(self.particles.x[i,:].squeeze(), landmark_mean)

            innovation = z - z_pred
            innovation[1] = wrap2Pi(innovation[1])

            new_weights[i] = multivariate_normal.pdf(innovation, np.array([0,0]), landmark_cov + self.measurement_covariance)

        # update weights using Bayes Rule
        self.particles.weights = np.multiply(self.particles.weights, new_weights)
        # normalize
        self.particles.weights = self.particles.weights/ np.sum(self.particles.weights)

        self.resampling()




    def resampling(self):
        # low variance resampling

        particles_x = self.particles.x
        particles_weights = self.particles.weights

        W = np.cumsum(particles_weights)

        r = np.random.rand(1) / self.num_particles
        # r = 0.5 / self.n
        j = 0 # one change, j = 1 in original, but then you can never select first particle, even if it has high weight
        for i in range(self.num_particles):

            u = r +  i/ self.num_particles
            while u > W[j]:
                j = j + 1
            self.particles.x[i,:] = particles_x[j, :]
            self.particles.weights[i] = 1 / self.num_particles
        





    


class particle_filter_landmark:

    def __init__(self, dimensions, inv_measurement_model, measurement_covariance, z, robot_pf, num_samples_per_robot_particle):

        self.dimensions = dimensions
        self.inv_measurement_model = inv_measurement_model
        self.num_particles = robot_pf.particles.x.shape[0] * num_samples_per_robot_particle
        self.measurement_covariance = measurement_covariance
        self.measurement_covariance_L = np.linalg.cholesky(measurement_covariance)
        self.num_samples_per_robot_particle = num_samples_per_robot_particle

        self.particles = particles(self.num_particles, self.dimensions)

        for i in range (robot_pf.particles.x.shape[0]):
            detected_landmark_location = inv_measurement_model(robot_pf.particles.x[i,:].squeeze(), z)

            for i in range (self.num_samples_per_robot_particle):
                sample_measurement_noise = self.measurement_covariance_L @ randn(z.shape[0],1)
                self.particles.x[i,:] = detected_landmark_location + sample_measurement_noise.squeeze()
                self.particles.weights[i] = 1/self.num_particles


    def update(self, z, robot_pf):

        likelihood_landmark_x = np.zeros([self.num_particles, self.dimensions])
        likelihood_landmark_weights = np.zeros([self.num_particles])
        for particle in robot_pf.particles.x:
            detected_landmark_location = self.inv_measurement_model(particle.squeeze(), z)

            for i in range (self.num_samples_per_robot_particle):
                sample_measurement_noise = self.measurement_covariance_L @ randn(z.shape[0],1)
                likelihood_landmark_x[i,:] = detected_landmark_location + sample_measurement_noise.squeeze()
                likelihood_landmark_weights[i] = 1/self.num_particles

        prior_x = self.particles.x
        prior_x_rolling = prior_x
        likelihood_x = likelihood_landmark_x
        likelihood_x_rolling = likelihood_x
        prior_updated_weight= np.zeros([self.num_particles])
        likelihood_updated_weight= np.zeros([self.num_particles])

        for i in range(self.num_particles):
            prior_updated_weight += np.sum((prior_x - likelihood_x_rolling) ** 2, axis = 1)
            likelihood_updated_weight += np.sum((likelihood_x - prior_x_rolling)**2, axis = 1)
            np.roll(prior_x_rolling, 1, axis = 0)
            np.roll(likelihood_x_rolling, 1, axis = 0)

        self.particles.weights = np.reciprocal( np.concatenate((prior_updated_weight,likelihood_updated_weight)) )
        self.particles.x = np.concatenate((prior_x, likelihood_x), axis = 0)

        return likelihood_landmark_x



        
    def resampling(self):
        # low variance resampling

        particles_x = self.particles.x
        particles_weights = self.particles.weights

        W = np.cumsum(particles_weights)

        r = np.random.rand(1) / self.num_particles
        # r = 0.5 / self.n
        j = 0 # one change, j = 1 in original, but then you can never select first particle, even if it has high weight
        for i in range(self.num_particles):

            u = r +  i/ self.num_particles
            while u > W[j]:
                j = j + 1
            self.particles.x[i,:] = particles_x[j, :]
            self.particles.weights[i] = 1 / self.num_particles
        




        
def calculateMeanCovLandmarkPF(particle_list):

    particle_array = np.array(particle_list)
    particle_average = np.sum(particle_array, axis=0) / particle_array.shape[0]

    zero_mean = particle_array - np.tile(particle_average, (particle_array.shape[0], 1))
    

    particle_covariance = zero_mean.T @ zero_mean / particle_array.shape[0]

    return particle_average, particle_covariance
    

    

