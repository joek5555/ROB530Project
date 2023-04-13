import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from utils import wrap2Pi, calculateMeanCovFromList

class particles():
    def __init__(self):
        self.state = []
        self.weight = []

class single_particle():
    def __init__(self):
        self.state = None
        self.weight = None

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
                self.particles.state.append((np.dot(initial_covariance_L, np.random.randn(initial_state.shape[0], 1)) + initial_state.reshape(-1,1)).reshape(-1))
                self.particles.weight.append(1/self.num_particles)


        else:
            self.particles.state = given_starting_particles
            self.num_particles = len(self.particles.state)
            self.particles.weight = (np.ones(len(self.particles.state)) / self.num_particles).tolist()

        


    def motion_step(self, u, dt):

        if self.process_input_size is not None and self.process_covariance is not None and self.process_model is not None:

            for i in range(self.num_particles):
                # generate noise to apply to the input
                sample_input_noise = self.process_covariance_L @ np.random.randn(self.process_input_size,1)

                self.particles.state[i] = self.process_model(self.particles.state[i], u + sample_input_noise.squeeze(), dt)
                #self.particles.state[i] = self.process_model(self.particles.state[i], u)
        else:
            print("ERROR: Atempting to run motion step when either process_input_size, process_covariance, or process_model is not defined")


    #update robot location | measurement of other robot or landmark
    #OG Kalman Filter
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

            #self.resampling()
            self.resampling(num_top_particles = 200)

        else:
            print("ERROR: Atempting to run measurement step when either measurement_covariance or measurement_model is not defined")


    #update landmark location | detection
    #Kalman filter
    def measurement_step_landmarks(self, detected_landmark_mean, detected_landmark_cov):

        new_weights = np.zeros([self.num_particles])
        for i in range(self.num_particles):

            innovation = detected_landmark_mean - self.particles.state[i]

            new_weights[i] = multivariate_normal.pdf(innovation, np.array([0,0]), detected_landmark_cov)

        # update weights using Bayes Rule
        new_weights = np.multiply(np.array(self.particles.weight), new_weights)
        # normalize
        new_weights = new_weights/ np.sum(new_weights)

        self.particles.weight = new_weights.tolist()

        #self.resampling()
        self.resampling(num_top_particles = 500)



    #update robot location | measurement of other robot or landmark  
    #brute force comparison (particle to particle)  
    def measurement_step_compare_particles(self,detected_landmark_particles):

        prior_state = np.array(self.particles.state)
        prior_state_rolling = prior_state
        likelihood_state = np.array(detected_landmark_particles)
        likelihood_state_rolling = likelihood_state
        prior_updated_weight= np.zeros(self.num_particles)
        likelihood_updated_weight= np.zeros(self.num_particles)

        for i in range(self.num_particles):
            prior_updated_weight += np.sum((prior_state - likelihood_state_rolling) ** 2, axis = 1)
            likelihood_updated_weight += np.sum((likelihood_state - prior_state_rolling)**2, axis = 1)
            prior_state_rolling = np.roll(prior_state_rolling, 1, axis = 0)
            likelihood_state_rolling = np.roll(likelihood_state_rolling, 1, axis = 0)

        particles_weight = np.reciprocal( np.concatenate((prior_updated_weight,likelihood_updated_weight)) )
        # normalize the weights 
        particles_weight = particles_weight / np.sum(particles_weight)
        self.particles.weights =  particles_weight.tolist()

        particles_x = np.concatenate((prior_state, likelihood_state), axis = 0)
        self.particles.x = particles_x.tolist()


        self.resampling()
        
    #update robot location | measurement of other robot or landmark
    #Piazza math
    def measurement_step_combine_gaussians(self, likelihood_mean, likelihood_covariance):
        prior_mean, prior_covariance = calculateMeanCovFromList(self.particles.state)
        posterior_mean = np.linalg.inv(np.linalg.inv(prior_covariance) + self.num_particles * np.linalg.inv(likelihood_covariance))@(
            np.linalg.inv(prior_covariance) @ prior_mean + self.num_particles * np.linalg.inv(likelihood_covariance) @ likelihood_mean)
        posterior_covariance = np.linalg.inv(np.linalg.inv(prior_covariance) + self.num_particles * np.linalg.inv(likelihood_covariance))
        
        #print("new run")
        #print(prior_mean)
        #print(prior_covariance)
        #print(likelihood_mean)
        #print(likelihood_covariance)
        #print(posterior_mean)
        #print(posterior_covariance)

        posterior_covariance_L = np.linalg.cholesky(posterior_covariance)

        self.particles.state.clear()
        self.particles.weight.clear()
        for i in range(self.num_particles):
            self.particles.state.append((np.dot(posterior_covariance_L, np.random.randn(posterior_mean.shape[0], 1)) + posterior_mean.reshape(-1,1)).reshape(-1))
            self.particles.weight.append(1/self.num_particles)


    


    def resampling(self, num_top_particles = None, tune_variance_factor = 0):
        # low variance resampling

        particles_x = np.array(self.particles.state)
        particles_weights = np.array(self.particles.weight)
        self.particles.state.clear()
        self.particles.weight.clear()
        #with np.printoptions(threshold=np.inf):
        #    print(np.sort(particles_weights))


        if num_top_particles is None:

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

            #with np.printoptions(threshold=np.inf):
            #    print(np.array(self.particles.state))
            #top_particles_mean, top_particles_variance = calculateMeanCovFromList(self.particles.state)
            

        else:
            particles_data = []

            W = np.cumsum(particles_weights)

            r = np.random.rand(1) / self.num_particles
            # r = 0.5 / self.n
            j = 0 # one change, j = 1 in original, but then you can never select first particle, even if it has high weight
            for i in range(self.num_particles):

                u = r +  i/ self.num_particles
                while u > W[j]:
                    j = j + 1
                particle = single_particle()
                #print(particles_x[j, :])
                particle.state = particles_x[j, :]   
                particle.weight = particles_weights[j]
                particles_data.append(particle)

            particles_data.sort(key=weight_sorting)
            #with np.printoptions(threshold=np.inf):
            #    print(np.array([particle.state for particle in particles_data]))

            #for i in range(0,100):
            #    print(particles_data[-i].state)
            #    print(particles_data[-i].weight)

            top_particles = particles_data[-num_top_particles:-1]
            top_particles_state = [particle.state for particle in top_particles]
            top_particles_mean, top_particles_variance = calculateMeanCovFromList(top_particles_state)

            if tune_variance_factor > 0:
                diagonal = top_particles_variance.diagonal()
                mean_variance = np.mean(diagonal)
                #greatest_variance = np.amax(diagonal)
                top_particles_variance = np.eye(top_particles_mean.shape[0]) * mean_variance * tune_variance_factor

            variance_L = np.linalg.cholesky(top_particles_variance)

            while len(top_particles_state) < self.num_particles:
                
                top_particles_state.append((np.dot(variance_L, np.random.randn(top_particles_mean.shape[0], 1)) + top_particles_mean.reshape(-1,1)).reshape(-1))
            
            self.particles.state = top_particles_state
            
            self.particles.weight = (np.ones(self.num_particles) * (1/self.num_particles)).tolist()



def weight_sorting(particle):
    return particle.weight 