import numpy as np
import matplotlib.pyplot as plt



def plot(groundtruth_x, groundtruth_y, landmarks_x, landmarks_y, robot1_system):

        robot1_particles = np.array(robot1_system.pf.particles.x)
        #print([particles])
        robot1_particles_x = robot1_particles[:,0]
        robot1_particles_y = robot1_particles[:,1]

        fig, ax = plt.subplots()
        ax.plot(groundtruth_x, groundtruth_y, 'r')
        ax.scatter(landmarks_x, landmarks_y, c='k', marker = "*")
        ax.scatter(robot1_particles_x, robot1_particles_y, s=1, c='b')

        for landmark_pf in robot1_system.detected_landmarks_pf:
            landmark_particles = np.array(landmark_pf.particles.x)

            landmark_particles_x = landmark_particles[:,0]
            landmakr_particles_y = landmark_particles[:,1]
            print(landmark_particles_x)
            print(landmakr_particles_y)
            ax.scatter(landmark_particles_x, landmakr_particles_y, s=1, c='g')

                
        ax.legend(['groundtruth','landmarks', 'particles'])

        plt.show()