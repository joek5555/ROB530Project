import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from utils import wrap2Pi

class particles():
    def __init__(self):
        self.state = []
        self.weight = []

class particle_filter:

    def __init__(self, process_model= None, process_covariance = None, process_input_size = None,
                  measurement_model = None, measurement_covariance = None,
                  initial_state = None, initial_covariance = None, 
                  num_particles = None, given_starting_particles = None):

        self.particles = particles()

        self.process_model = process_model
        self.process_covariance = process_covariance
        if self.process_covariance is not None:
            self.process_covariance_L = np.linalg.cholesky(self.process_covariance)
        else:
            self.process_covariance_L = None
        self.process_input_size = process_input_size

        self.measurement_model = measurement_model
        self.measurement_covariance = measurement_covariance
        #self.measurement_covariance_L = np.linalg.cholesky(self.measurement_covariance)
        

        # two ways to initialize the particles in the pf
        # can provide initial state, initial covariance, and num particles, then particles are calculated
        # or can just provide list of starting particles
        if num_particles is not None:
            self.num_particles = num_particles
            initial_covariance_L = np.linalg.cholesky(initial_covariance)

            for i in range(self.num_particles):
                self.particles.state.append((np.dot(initial_covariance_L, np.random.randn(initial_state.shape[0], 1)) + initial_state).reshape(-1))
                self.particles.weight.append(1/self.num_particles)


        else:
            self.particles.state = given_starting_particles
            self.num_particles = len(self.particles.state)
            self.particles.weight = (np.ones(len(self.particles.state)) / self.num_particles).tolist()

        


    def motion_step(self, u):

        if self.process_input_size is not None and self.process_covariance is not None and self.process_model is not None:

            for i in range(self.num_particles):
                # generate noise to apply to the input
                sample_input_noise = self.process_covariance_L @ np.random.randn(self.process_input_size,1)

                self.particles.state[i] = self.process_model(self.particles.state[i], u + sample_input_noise.squeeze())
                #self.particles.state[i] = self.process_model(self.particles.state[i], u)
        else:
            print("ERROR: Atempting to run motion step when either process_input_size, process_covariance, or process_model is not defined")

    def measurement_step(self, z, landmark_mean, landmark_cov):
        
        if self.measurement_model is not None and self.measurement_covariance is not None:

            new_weights = np.zeros([self.num_particles])
            for i in range(self.num_particles):

                z_pred = self.measurement_model(self.particles.state[i], landmark_mean)

                innovation = z - z_pred
                innovation[1] = wrap2Pi(innovation[1])

                new_weights[i] = multivariate_normal.pdf(innovation, np.array([0,0]), landmark_cov + self.measurement_covariance)

            # update weights using Bayes Rule
            new_weights = np.multiply(np.array(self.particles.weight), new_weights)
            # normalize
            new_weights = new_weights/ np.sum(new_weights)

            self.particles.weight = new_weights.tolist()

            self.resampling()

        else:
            print("ERROR: Atempting to run measurement step when either measurement_covariance or measurement_model is not defined")




    def resampling(self):
        # low variance resampling

        particles_x = np.array(self.particles.state)
        particles_weights = np.array(self.particles.weight)
        self.particles.state.clear()
        self.particles.weight.clear()

        W = np.cumsum(particles_weights)

        r = np.random.rand(1) / self.num_particles
        # r = 0.5 / self.n
        j = 0 # one change, j = 1 in original, but then you can never select first particle, even if it has high weight
        for i in range(self.num_particles):

            u = r +  i/ self.num_particles
            while u > W[j]:
                j = j + 1
            self.particles.state.append(particles_x[j, :])
            self.particles.weight.append( 1 / self.num_particles)