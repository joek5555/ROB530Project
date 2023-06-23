# ROB 530 Final Project:Designing a Multi-Agent Localization Algorithm w/ Selective Communication

A video description of the project can be found [here](https://www.youtube.com/watch?v=uGdUyuPvICg).

This repository is designed to help address the problem of localizing robots in an unknown environment. Multiple robots starting from a known position with respect to one another will explore a new environment, trying to localize themselves and map out features in the environment. These features, or landmarks, in the environment can be classified, but it is not known ahead of time where each landmark is. 

The UTIAS dataset is used to run tests on our implementation. This dataset contains the odometry, measurement, and groundtruth values of five robots driving in an environment. Thus, it matches with our problem statement. 

Our implementation uses particle filters on each robot to track the robot's position. Additionally, each robot has a sperate particle filter to track landmark locations. The first time a robot sees a landmark, the robot will initialize the particle filter for that particular landmark ID. The robot will not use this measurement to update its own position the first time. Later, if the robot measures the same landmark, it will use that measurement to update the landmark's particle filter, reducing the variance of the robot's estimation of where that landmark is. Next, the robot can update its own position using the landmark position.

The repository contains code to run a test dataset in the PF_landmarks folder. The code to run the UTIAS dataset is found in PF_landmarks_real_data. 
The run.py file contains all the code to initialize the robots and run a test through the data.
The settings.yaml file specifies the number of robots to run, how long to run the test, and some noise parameters.
The robot_system.py file contains a robot class, which holds all the variables associated with each robot. 
The models.py file contains the motion models, measurement models, and inverse measurement models.
The PF.py file contains the particle filter class, which holds all functions required to run the motion step, measurement step, and resampling step. 
The utils.py file contains helper function to read the data, plot the data, and comupte calculations such as the mean and varaince of a list of particles.

To run the algorithm, please launch the run.py file in the ```PF_landmarks_reald_data``` directory. For the UTIAS dataset, you can use the -d tag to specity the maximum distance the robot can detect measurements. We found that limiting the robot to only detect landmarks that are fairly close (6.0 meters away) lead to good results.

For information on what flags can be provided, run ```python3 run.py --help```.


![animated_from_images](https://user-images.githubusercontent.com/92048856/233205552-24041044-36d6-454f-ab79-2f9ba470bf5d.gif)

