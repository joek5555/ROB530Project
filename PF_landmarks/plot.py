import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse




def plot(groundtruth_x, groundtruth_y, landmarks_x, landmarks_y, robot1_system):

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

            plot_covariance(ax, landmark_pf.mean, landmark_pf.cov)

                
        ax.legend(['groundtruth','landmarks', 'particles'])

        plt.savefig("bruh.png")


def plot_covariance(ax, mean, cov):
    xy = (mean[0], mean[1])
    
    a = cov[0, 0]
    b = cov[0, 1]
    c = cov[1, 1]

    term_1 = (a + c) / 2
    term_2 = np.sqrt((((a - c) / 2) ** 2) + b ** 2)

    lambda_1 = term_1 + term_2
    lambda_2 = term_1 - term_2

    if b == 0:
        if a >= c:
            theta = 0
        else:
            theta = 45
    else:
        theta = np.arctan2(lambda_1 - a, b)

    ellipse = Ellipse(xy, width=2 * lambda_1, height=2 * lambda_2, angle=theta, fill=False, color='red')
    # ax.add_patch(ellipse)


# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.ion()
  
# mean = np.array([0.25, 0.5])
# cov = np.array([[0.1, 0], [0, 0.25]])
# plot_covariance(ax, mean, cov)
# plt.show()
# plt.pause(1)

# mean = np.array([0.6, 0.8])
# cov = np.array([[0.1, 0], [0, 0.25]])
# plot_covariance(ax, mean, cov)

# plt.ioff()
# plt.show()