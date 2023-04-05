import numpy as np
from numpy.random import randn, rand
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from utilities import wrap2Pi

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
            #self.particles.x.append(self.initial_state)
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
        





    


class particle_filter_landmark:

    def __init__(self, inv_measurement_model, measurement_covariance, z, robot_pf, num_samples_per_robot_particle):

        self.inv_measurement_model = inv_measurement_model
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


    def update(self, z, robot_pf):

        likelihood_landmark_x = []
        likelihood_landmark_weights = []
        for particle in robot_pf.particles.x:
            detected_landmark_location = self.inv_measurement_model(particle.squeeze(), z)
            

            for i in range (self.num_samples_per_robot_particle):
                sample_measurement_noise = self.measurement_covariance_L @ randn(z.shape[0],1)
                likelihood_landmark_x.append(detected_landmark_location + sample_measurement_noise.squeeze())
                likelihood_landmark_weights.append(1/self.num_particles)

        prior_x = np.array(self.particles.x)
        prior_x_rolling = prior_x
        likelihood_x = np.array(likelihood_landmark_x)
        likelihood_x_rolling = likelihood_x
        prior_updated_weight= np.zeros(self.num_particles)
        likelihood_updated_weight= np.zeros(self.num_particles)

        for i in range(self.num_particles):
            prior_updated_weight += np.sum((prior_x - likelihood_x_rolling) ** 2, axis = 1)
            likelihood_updated_weight += np.sum((likelihood_x - prior_x_rolling)**2, axis = 1)
            prior_x_rolling = np.roll(prior_x_rolling, 1, axis = 0)
            likelihood_x_rolling = np.roll(likelihood_x_rolling, 1, axis = 0)

        particles_weight = np.reciprocal( np.concatenate((prior_updated_weight,likelihood_updated_weight)) )
        self.particles.weights =  particles_weight.tolist()
        #particles_weight = np.sort(particles_weight)
        particles_x = np.concatenate((prior_x, likelihood_x), axis = 0)
        self.particles.x = particles_x.tolist()
        #with np.printoptions(threshold=np.inf):
        #    print(np.sort(particles_weight))

        return likelihood_landmark_x



        
    def resampling(self):
        # low variance resampling

        particles_weight = np.array(self.particles.weights)
        particles_x = np.array(self.particles.x)
        self.particles.x.clear()
        self.particles.weights.clear()


        W = np.cumsum(particles_weight)

        r = np.random.rand(1) / self.num_particles
        # r = 0.5 / self.n
        j = 0 # one change, j = 1 in original, but then you can never select first particle, even if it has high weight
        for i in range(self.num_particles):

            u = r +  i/ self.num_particles
            while u > W[j]:
                j = j + 1
            self.particles.x.append(particles_x[j, :])
            self.particles.weights.append( 1 / self.num_particles)
        




        
def calculateMeanCovLandmarkPF(particle_list):

    particle_array = np.array(particle_list)
    particle_average = np.sum(particle_array, axis=0) / particle_array.shape[0]

    zero_mean = particle_array - np.tile(particle_average, (particle_array.shape[0], 1))
    

    particle_covariance = zero_mean.T @ zero_mean / particle_array.shape[0]

    return particle_average, particle_covariance
    

    

