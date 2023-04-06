import numpy as np
import matplotlib.pyplot as plt



def plot(robot1_groundtruth_x, robot1_groundtruth_y, robot2_groundtruth_x, robot2_groundtruth_y,
        landmarks_x, landmarks_y, robot1_system, robot2_system, additional_landmark = None, main_robot = 1, timestep = 0):

        robot1_particles = np.array(robot1_system.pf.particles.x)
        robot2_particles = np.array(robot2_system.pf.particles.x)

        robot1_particles_x = robot1_particles[:,0]
        robot1_particles_y = robot1_particles[:,1]

        robot2_particles_x = robot2_particles[:,0]
        robot2_particles_y = robot2_particles[:,1]        

        fig, ax = plt.subplots()
        ax.plot(robot1_groundtruth_x, robot1_groundtruth_y, 'r')
        ax.plot(robot2_groundtruth_x, robot2_groundtruth_y, 'y')
        ax.scatter(landmarks_x, landmarks_y, c='k', marker = "*")
        ax.scatter(robot1_particles_x, robot1_particles_y, s=1, c='b')
        ax.scatter(robot2_particles_x, robot2_particles_y, s=1, c='c')

        if main_robot == 1:

            for landmark_id, landmark_pf in robot1_system.detected_landmarks.items():

                landmark_particles = np.array(landmark_pf.particles.x)

                landmark_particles_x = landmark_particles[:,0]
                landmark_particles_y = landmark_particles[:,1]
                ax.scatter(landmark_particles_x, landmark_particles_y, s=1, c='#420b4d')
            
            if additional_landmark != None:
                additional_landmark_array = np.array(additional_landmark)
                additional_landmark_x = additional_landmark_array[:,0]
                additional_landmark_y = additional_landmark_array[:,1]
                ax.scatter(additional_landmark_x, additional_landmark_y, s=1, c='#cf77e0')

        elif main_robot == 2:
            for landmark_id, landmark_pf in robot2_system.detected_landmarks.items():

                landmark_particles = np.array(landmark_pf.particles.x)

                landmark_particles_x = landmark_particles[:,0]
                landmark_particles_y = landmark_particles[:,1]
                ax.scatter(landmark_particles_x, landmark_particles_y, s=1, c='#064f1a')
            
            if additional_landmark != None:
                additional_landmark_array = np.array(additional_landmark)
                additional_landmark_x = additional_landmark_array[:,0]
                additional_landmark_y = additional_landmark_array[:,1]
                ax.scatter(additional_landmark_x, additional_landmark_y, s=1, c='#79d191')

                
        ax.legend(['robot1_gt', 'robot2_gt','landmarks', 'robot1_particles', 'robot2_particles', 'prior', 'likelihood'])
        plt.title("robot: " + str(main_robot) + ", timestep: " +  str(timestep))
        if additional_landmark == None:
            plt.savefig(f"PF_landmarks/saved_images/robot_" + str(main_robot) + "_t_" +  str(timestep) + ".png")
        else:
            plt.savefig(f"PF_landmarks/saved_images/robot_" + str(main_robot) + "_t_" +  str(timestep) + "_pre_update.png")
        #plt.show()
        plt.close()


def wrap2Pi(input):
    phases =  (( -input + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0

    return phases