import numpy as np
from numpy.random import randn, rand
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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
            #self.particles.x.append(np.dot(initial_covariance_L, randn(len(initial_state), 1)) + initial_state)
            self.particles.x.append(self.initial_state)
            self.particles.weights.append(1/self.num_particles)

    def motion_step(self, u):

        for i in range(self.num_particles):
            # generate noise to apply to the input
            sample_input_noise = self.process_covariance_L @ randn(self.process_input_size,1)
            #print(sample_input_noise)

            self.particles.x[i] = self.process_model(self.particles.x[i], u + sample_input_noise.squeeze())
            #self.particles.x[i] = self.process_model(self.particles.x[i], u)






    


class particle_filter_landmark:

    def __init__(self, inv_measurement_model, measurement_covariance, z, robot_pf, num_samples_per_robot_particle):

        self.measurement_model = inv_measurement_model
        self.particles = particles()
        self.num_particles = len(robot_pf.particles.x) * num_samples_per_robot_particle
        self.measurement_covariance = measurement_covariance
        self.measurement_covariance_L = np.linalg.cholesky(measurement_covariance)
        self.num_samples_per_robot_particle = num_samples_per_robot_particle

        for particle in robot_pf.particles.x:
            detected_landmark_location = inv_measurement_model(particle.squeeze(), z)

            for i in range (self.num_samples_per_robot_particle):
                sample_measurement_noise = self.measurement_covariance_L @ randn(z.shape[0],1)
                self.particles.x.append(detected_landmark_location + sample_measurement_noise.squeeze())
                self.particles.weights.append(1/self.num_particles)
    
        self.mean, self.cov = calculateMeanCovLandmarkPF(self.particles.x)


    def update(self, z, robot_pf):
        for particle in robot_pf.particles.x:
            detected_landmark_location = self.measurement_model(particle.squeeze(), z)
            updated_landmarks = particles()

            for i in range (self.num_samples_per_robot_particle):
                sample_measurement_noise = self.measurement_covariance_L @ randn(z.shape[0],1)
                updated_landmarks.x.append(detected_landmark_location + sample_measurement_noise.squeeze())
                updated_landmarks.weights.append(1/self.num_particles)

        #calculate mean and cov for current
        curr_mean, curr_cov = calculateMeanCovLandmarkPF(self.particles.x)
        #calculate mean and cov for updated
        updated_mean, updated_cov = calculateMeanCovLandmarkPF(updated_landmarks.x)
        #combine
        combined_mean, combined_cov = combine_gaussians(curr_mean, curr_cov, updated_mean, updated_cov)
        self.mean = combined_mean
        self.cov = combined_cov
        #low variance resampling to sample 1000 points from this distribution
            
        
def calculateMeanCovLandmarkPF(particle_list):
    assert(len(particle_list) != 0)

    particle_array = np.array(particle_list)
    
    particle_average = np.sum(particle_array, axis=0) / particle_array.shape[0]


    zero_mean = particle_array - np.tile(particle_average, (particle_array.shape[0], 1))

    particle_covariance = zero_mean.T @ zero_mean / particle_array.shape[0]

    return particle_average, particle_covariance

def combine_gaussians(mu_1, sigma_1, mu_2, sigma_2):
    combined_mu = mu_2 * (sigma_1**2) / (sigma_1**2 + sigma_2**2) + mu_1 * (sigma_2**2) / (sigma_1**2 + sigma_2**2)
    print(sigma_1, sigma_2)
    combined_cov = sigma_1**2*sigma_2**2/(sigma_1**2 + sigma_2**2)
    return combined_mu, combined_cov
    

    

        