import numpy as np
import matplotlib.pyplot as plt



def plot(groundtruth_x, groundtruth_y, landmarks_x, landmarks_y, robot1_system, additional_landmark = None):

        robot1_particles = np.array(robot1_system.pf.particles.x)
        #print([particles])
        robot1_particles_x = robot1_particles[:,0]
        robot1_particles_y = robot1_particles[:,1]

        fig, ax = plt.subplots()
        ax.plot(groundtruth_x, groundtruth_y, 'r')
        ax.scatter(landmarks_x, landmarks_y, c='k', marker = "*")
        ax.scatter(robot1_particles_x, robot1_particles_y, s=1, c='b')

        for landmark_id, landmark_pf in robot1_system.detected_landmarks.items():
            if landmark_id == 6:
                color = 'g'
            elif landmark_id ==7:
                 color = 'r'
            landmark_particles = np.array(landmark_pf.particles.x)

            landmark_particles_x = landmark_particles[:,0]
            landmark_particles_y = landmark_particles[:,1]
            ax.scatter(landmark_particles_x, landmark_particles_y, s=1, c=color)
        
        if additional_landmark != None:
            additional_landmark_array = np.array(additional_landmark)
            additional_landmark_x = additional_landmark_array[:,0]
            additional_landmark_y = additional_landmark_array[:,1]
            ax.scatter(additional_landmark_x, additional_landmark_y, s=1, c='m')

                
        ax.legend(['groundtruth','landmarks', 'particles'])

        plt.show()


def wrap2Pi(input):
    phases =  (( -input + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0

    return phases