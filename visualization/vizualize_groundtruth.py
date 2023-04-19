import numpy as np
import os
import matplotlib.pyplot as plt
import imageio

# parameters to set before running
seconds_to_run = 240
plot_connections = False


landmark_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Landmark_Groundtruth.dat'))
robot1_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot1_Groundtruth.dat'))
robot2_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot2_Groundtruth.dat'))
robot3_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot3_Groundtruth.dat'))
robot4_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot4_Groundtruth.dat'))
robot5_groundtruth_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot5_Groundtruth.dat'))

robot1_measurement_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'UTIAS-dataset/MRCLAM_Dataset1', 'Robot1_Measurement_x.dat'))


landmark_groundtruth = np.loadtxt(landmark_groundtruth_path)
robot1_groundtruth = np.loadtxt(robot1_groundtruth_path)
robot2_groundtruth = np.loadtxt(robot2_groundtruth_path)
robot3_groundtruth = np.loadtxt(robot3_groundtruth_path)
robot4_groundtruth = np.loadtxt(robot4_groundtruth_path)
robot5_groundtruth = np.loadtxt(robot5_groundtruth_path)

robot1_measurements = np.loadtxt(robot1_measurement_path)


landmark_x = landmark_groundtruth[:,1]
landmark_y = landmark_groundtruth[:,2]


fig, ax = plt.subplots()
ax.scatter(landmark_x, landmark_y, c='k')

end = seconds_to_run * 40
ax.plot(
    robot1_groundtruth[0:end,1], robot1_groundtruth[0:end,2], 'r',
    robot2_groundtruth[0:end,1], robot2_groundtruth[0:end,2], 'y',
    robot3_groundtruth[0:end,1], robot3_groundtruth[0:end,2], 'g',
    robot4_groundtruth[0:end,1], robot4_groundtruth[0:end,2], 'b',
    robot5_groundtruth[0:end,1], robot5_groundtruth[0:end,2], 'm')
ax.legend(['landmarks', 'robot1', 'robot2', 'robot3', 'robot4', 'robot5'])
ax.scatter(robot1_groundtruth[0,1], robot1_groundtruth[0,2], marker='s', c='r')
ax.scatter(robot2_groundtruth[0,1], robot2_groundtruth[0,2], marker='s', c='y')
ax.scatter(robot3_groundtruth[0,1], robot3_groundtruth[0,2], marker='s', c='g')
ax.scatter(robot4_groundtruth[0,1], robot4_groundtruth[0,2], marker='s', c='b')
ax.scatter(robot5_groundtruth[0,1], robot5_groundtruth[0,2], marker='s', c='m')

ax.set(xlabel='x', ylabel='y',
       title='Robot groundtruth')

robot1_end_time = robot1_groundtruth[end, 0]
robot1_measurement_time = robot1_measurements[0,0]
measure_index = 0
groundtruth_index = 0
if plot_connections:
    while robot1_measurement_time <= robot1_end_time:
        while robot1_groundtruth[groundtruth_index,0] - robot1_measurements[measure_index,0] < -0.025:
            groundtruth_index +=1
        if groundtruth_index > end:
            break

        if robot1_measurements[measure_index, 1] == 2:
            measurement_endpoint = np.array([robot2_groundtruth[groundtruth_index,1], robot2_groundtruth[groundtruth_index,2]])
        elif robot1_measurements[measure_index, 1] == 3:
            measurement_endpoint = np.array([robot3_groundtruth[groundtruth_index,1], robot3_groundtruth[groundtruth_index,2]])
        elif robot1_measurements[measure_index, 1] == 4:
            measurement_endpoint = np.array([robot4_groundtruth[groundtruth_index,1], robot4_groundtruth[groundtruth_index,2]])
        elif robot1_measurements[measure_index, 1] == 5:
            measurement_endpoint = np.array([robot5_groundtruth[groundtruth_index,1], robot5_groundtruth[groundtruth_index,2]])
        else:
            landmark_index = int(robot1_measurements[measure_index, 1] - 6)
            measurement_endpoint = np.array([landmark_groundtruth[landmark_index,1], landmark_groundtruth[landmark_index,2]])

        x_values = [robot1_groundtruth[groundtruth_index,1] ,measurement_endpoint[0]]
        y_values = [robot1_groundtruth[groundtruth_index,2] ,measurement_endpoint[1]]
        ax.plot(x_values, y_values, 'k', linestyle="--", linewidth=1)
        measure_index += 1
        #print(robot1_measurements.size)
        if measure_index >= robot1_measurements.shape[0]:
            break

plt.show()

"""


seconds_to_run = 5
end = seconds_to_run * 20
frames = []
for i in range(end):
    plt.scatter(landmark_x, landmark_y, c='k')
    plt.scatter(robot1_groundtruth[i*2,1], robot1_groundtruth[i*2,2], s=1, c='r')
    plt.scatter(robot2_groundtruth[i*2,1], robot2_groundtruth[i*2,2], s=1, c='y')
    plt.scatter(robot3_groundtruth[i*2,1], robot3_groundtruth[i*2,2], s=1, c='g')
    plt.scatter(robot4_groundtruth[i*2,1], robot4_groundtruth[i*2,2], s=1, c='b')
    plt.scatter(robot5_groundtruth[i*2,1], robot5_groundtruth[i*2,2], s=1, c='m')
    #plt.set(xlabel='x', ylabel='y',
    #   title='Robot groundtruth first 30 s')
    
    figfolder = "fig"
    if not os.path.exists(figfolder):
        os.makedirs(figfolder)
    plt.savefig(figfolder + f"/gt_image_{i+1}.png")    
    plt.pause(0.1)

    frames.append(imageio.imread(f"fig/gt_image_{i+1}.png"))


imageio.mimsave('animation.gif', frames, fps=40)

"""