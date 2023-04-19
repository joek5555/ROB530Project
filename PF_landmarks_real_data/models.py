import numpy as np
from utils import wrap2Pi


def process_model_noise(u, alphas):
    output = np.array([[alphas[0]*u[0]**2+alphas[1]*u[1]**2,0,0], 
        [0,alphas[2]*u[0]**2+alphas[3]*u[1]**2,0],
        [0,0,alphas[4]*u[0]**2+alphas[5]*u[1]**2]])
    return output

def process_model(x,u, dt):

    output = np.zeros(3)
    output[0] = x[0] + (-u[0] / u[1] * np.sin(x[2]) + u[0] / u[1] * np.sin(x[2] + u[1] * dt))
    output[1] = x[1] + ( u[0] / u[1] * np.cos(x[2]) - u[0] / u[1] * np.cos(x[2] + u[1] * dt))
    output[2] = x[2] + u[1]*dt + u[2]*dt
    return output

def inv_measurement_model(x,z):
    output = np.zeros(2)
    output[0] = x[0] + z[0]* np.cos(x[2] + z[1])
    output[1] = x[1] + z[0] * np.sin(x[2] + z[1])
    return output

def measurement_model(x,landmark):
    output = np.zeros(2)
    output[0] = np.sqrt((landmark[1] - x[1])**2 + (landmark[0] - x[0])**2)
    output[1] = wrap2Pi(np.arctan2(landmark[1] - x[1], landmark[0] - x[0]) - x[2])
    return output

#def landmark_measurement_model(landmark1, landmark2):
#    output = (landmark1 - landmark2).squeeze()
#    return output

def landmark_measurement_model(landmark1, landmark2):
    output = (landmark1 - landmark2).squeeze()
    return output