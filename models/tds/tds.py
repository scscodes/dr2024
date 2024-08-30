import math
def reward_function(params):

    def dist_2_points(x1, x2, y1, y2):
        return abs(abs(x1-x2)**2 + abs(y1-y2)**2)**0.5

    def closest_2_racing_points_index(racing_coords, car_coords):

        # Calculate all distances to racing points
        distances = []
        for i in range(len(racing_coords)):
            distance = dist_2_points(x1=racing_coords[i][0], x2=car_coords[0],
                                     y1=racing_coords[i][1], y2=car_coords[1])
            distances.append(distance)

        # Get index of the closest racing point
        closest_index = distances.index(min(distances))

        # Get index of the second closest racing point
        distances_no_closest = distances.copy()
        distances_no_closest[closest_index] = 999
        second_closest_index = distances_no_closest.index(
            min(distances_no_closest))

        return [closest_index, second_closest_index]

    def dist_to_racing_line(closest_coords, second_closest_coords, car_coords):

        # Calculate the distances between 2 closest racing points
        a = abs(dist_2_points(x1=closest_coords[0],
                              x2=second_closest_coords[0],
                              y1=closest_coords[1],
                              y2=second_closest_coords[1]))

        # Distances between car and closest and second closest racing point
        b = abs(dist_2_points(x1=car_coords[0],
                              x2=closest_coords[0],
                              y1=car_coords[1],
                              y2=closest_coords[1]))
        c = abs(dist_2_points(x1=car_coords[0],
                              x2=second_closest_coords[0],
                              y1=car_coords[1],
                              y2=second_closest_coords[1]))

        # Calculate distance between car and racing line (goes through 2 closest racing points)
        # try-except in case a=0 (rare bug in DeepRacer)
        try:
            distance = abs(-(a**4) + 2*(a**2)*(b**2) + 2*(a**2)*(c**2) -
                           (b**4) + 2*(b**2)*(c**2) - (c**4))**0.5 / (2*a)
        except:
            distance = b

        return distance

    # Calculate which one of the closest racing points is the next one and which one the previous one
    def next_prev_racing_point(closest_coords, second_closest_coords, car_coords, heading):

        # Virtually set the car more into the heading direction
        heading_vector = [math.cos(math.radians(
            heading)), math.sin(math.radians(heading))]
        new_car_coords = [car_coords[0]+heading_vector[0],
                          car_coords[1]+heading_vector[1]]

        # Calculate distance from new car coords to 2 closest racing points
        distance_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                    x2=closest_coords[0],
                                                    y1=new_car_coords[1],
                                                    y2=closest_coords[1])
        distance_second_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                           x2=second_closest_coords[0],
                                                           y1=new_car_coords[1],
                                                           y2=second_closest_coords[1])

        if distance_closest_coords_new <= distance_second_closest_coords_new:
            next_point_coords = closest_coords
            prev_point_coords = second_closest_coords
        else:
            next_point_coords = second_closest_coords
            prev_point_coords = closest_coords

        return [next_point_coords, prev_point_coords]

    def racing_direction_diff(closest_coords, second_closest_coords, car_coords, heading):

        # Calculate the direction of the center line based on the closest waypoints
        next_point, prev_point = next_prev_racing_point(closest_coords,
                                                        second_closest_coords,
                                                        car_coords,
                                                        heading)

        # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
        track_direction = math.atan2(
            next_point[1] - prev_point[1], next_point[0] - prev_point[0])

        # Convert to degree
        track_direction = math.degrees(track_direction)

        # Calculate the difference between the track direction and the heading direction of the car
        direction_diff = abs(track_direction - heading)
        if direction_diff > 180:
            direction_diff = 360 - direction_diff

        return direction_diff

    # Gives back indexes that lie between start and end index of a cyclical list
    # (start index is included, end index is not)
    def indexes_cyclical(start, end, array_len):

        if end < start:
            end += array_len

        return [index % array_len for index in range(start, end)]

    # Calculate how long car would take for entire lap, if it continued like it did until now
    def projected_time(first_index, closest_index, step_count, times_list):

        # Calculate how much time has passed since start
        current_actual_time = (step_count-1) / 15

        # Calculate which indexes were already passed
        indexes_traveled = indexes_cyclical(first_index, closest_index, len(times_list))

        # Calculate how much time should have passed if car would have followed optimals
        current_expected_time = sum([times_list[i] for i in indexes_traveled])

        # Calculate how long one entire lap takes if car follows optimals
        total_expected_time = sum(times_list)

        # Calculate how long car would take for entire lap, if it continued like it did until now
        try:
            projected_time = (current_actual_time/current_expected_time) * total_expected_time
        except:
            projected_time = 9999

        return projected_time

    #################### RACING LINE ######################


    # Each row: [x,y,speed,timeFromPreviousPoint]
    racing_track = [[5.04772, 0.65863, 3.75, 0.05491],
                    [5.04771, 0.84177, 3.75, 0.04884],
                    [5.04769, 1.00937, 3.68753, 0.04545],
                    [5.04768, 1.17083, 3.34137, 0.04832],
                    [5.04602, 1.46946, 3.07352, 0.09716],
                    [5.0392, 1.76713, 2.85639, 0.10424],
                    [5.02428, 2.06235, 2.67424, 0.11053],
                    [4.99854, 2.35347, 2.52031, 0.11596],
                    [4.95962, 2.63886, 2.38673, 0.12068],
                    [4.90545, 2.91697, 2.23236, 0.12692],
                    [4.83429, 3.18622, 2.23236, 0.12475],
                    [4.74463, 3.44499, 2.23236, 0.12268],
                    [4.63515, 3.6915, 2.23236, 0.12083],
                    [4.5047, 3.9238, 2.23236, 0.11934],
                    [4.35225, 4.13955, 2.23236, 0.11834],
                    [4.17621, 4.33521, 2.32254, 0.11333],
                    [3.97964, 4.51101, 2.42228, 0.10887],
                    [3.76509, 4.66727, 2.53098, 0.10487],
                    [3.53475, 4.80441, 2.65251, 0.10107],
                    [3.29054, 4.92295, 2.78946, 0.09732],
                    [3.03423, 5.0235, 2.94464, 0.0935],
                    [2.76746, 5.10684, 3.12722, 0.08937],
                    [2.49181, 5.17397, 3.34547, 0.0848],
                    [2.20878, 5.22611, 3.61283, 0.07966],
                    [1.91981, 5.26474, 3.75, 0.07774],
                    [1.62622, 5.2916, 3.75, 0.07862],
                    [1.32924, 5.30863, 3.75, 0.07933],
                    [1.02992, 5.31797, 3.75, 0.07986],
                    [0.72916, 5.32186, 3.75, 0.08021],
                    [0.42768, 5.32264, 3.72712, 0.08089],
                    [0.1261, 5.3222, 3.42182, 0.08813],
                    [-0.17493, 5.3179, 3.1798, 0.09468],
                    [-0.47491, 5.30716, 2.98194, 0.10066],
                    [-0.77317, 5.28744, 2.81647, 0.10613],
                    [-1.06881, 5.25635, 2.67302, 0.11121],
                    [-1.36063, 5.2116, 2.54602, 0.11596],
                    [-1.64713, 5.15116, 2.43241, 0.12038],
                    [-1.92643, 5.07329, 2.32986, 0.12445],
                    [-2.19632, 4.97663, 2.23657, 0.12818],
                    [-2.45429, 4.8603, 2.15085, 0.13157],
                    [-2.69756, 4.72392, 2.06745, 0.13489],
                    [-2.92317, 4.56766, 1.99241, 0.13775],
                    [-3.12811, 4.39227, 1.92101, 0.14042],
                    [-3.30935, 4.19908, 1.85091, 0.14312],
                    [-3.46403, 3.99001, 1.7853, 0.14567],
                    [-3.58945, 3.76746, 1.72175, 0.14837],
                    [-3.68298, 3.5343, 1.66108, 0.15125],
                    [-3.74231, 3.29386, 1.66108, 0.14909],
                    [-3.7652, 3.04998, 1.66108, 0.14746],
                    [-3.74943, 2.8071, 1.66108, 0.14653],
                    [-3.6929, 2.57048, 1.66108, 0.14645],
                    [-3.59362, 2.34668, 1.66108, 0.14739],
                    [-3.45025, 2.14422, 1.90615, 0.13015],
                    [-3.27506, 1.96322, 2.02579, 0.12435],
                    [-3.07274, 1.8049, 2.16569, 0.11862],
                    [-2.84746, 1.66958, 2.33742, 0.11243],
                    [-2.60309, 1.55668, 2.55291, 0.10544],
                    [-2.34323, 1.46477, 2.84055, 0.09704],
                    [-2.07125, 1.39147, 3.24885, 0.0867],
                    [-1.79029, 1.3335, 2.84231, 0.10093],
                    [-1.50327, 1.28682, 2.52789, 0.11503],
                    [-1.21286, 1.24681, 2.29732, 0.12761],
                    [-0.92767, 1.20238, 2.11897, 0.13621],
                    [-0.6484, 1.14803, 1.97295, 0.14421],
                    [-0.37838, 1.07929, 1.8508, 0.15054],
                    [-0.1211, 0.99262, 1.74531, 0.15555],
                    [0.11992, 0.8856, 1.65465, 0.15938],
                    [0.34105, 0.75685, 1.57448, 0.16252],
                    [0.53856, 0.60607, 1.5, 0.16565],
                    [0.70847, 0.43392, 1.5, 0.16126],
                    [0.84653, 0.24202, 1.5, 0.1576],
                    [0.94811, 0.03318, 1.5, 0.15483],
                    [1.00828, -0.1884, 1.5, 0.15307],
                    [1.02191, -0.41654, 1.5, 0.15237],
                    [0.98399, -0.64221, 1.67051, 0.13698],
                    [0.90516, -0.85861, 1.74471, 0.13201],
                    [0.78942, -1.06137, 1.82786, 0.12772],
                    [0.64051, -1.2472, 1.91939, 0.12407],
                    [0.46183, -1.4136, 2.02328, 0.12067],
                    [0.25664, -1.55865, 2.14561, 0.11711],
                    [0.02822, -1.68114, 2.29214, 0.11308],
                    [-0.22013, -1.78058, 2.47281, 0.10818],
                    [-0.48502, -1.85738, 2.70242, 0.10206],
                    [-0.76308, -1.91292, 3.01059, 0.09418],
                    [-1.05101, -1.94965, 3.35217, 0.08659],
                    [-1.34585, -1.97083, 2.97339, 0.09941],
                    [-1.64498, -1.98058, 2.69721, 0.11096],
                    [-1.94628, -1.98353, 2.48281, 0.12136],
                    [-2.24452, -1.99471, 2.3097, 0.12922],
                    [-2.53743, -2.01827, 2.16679, 0.13562],
                    [-2.82243, -2.05778, 2.03927, 0.14109],
                    [-3.09675, -2.11599, 1.92825, 0.14543],
                    [-3.35753, -2.19479, 1.88265, 0.1447],
                    [-3.60183, -2.29526, 1.8092, 0.14601],
                    [-3.82672, -2.41766, 1.74164, 0.14701],
                    [-4.02924, -2.56156, 1.67463, 0.14835],
                    [-4.20622, -2.72599, 1.67463, 0.14426],
                    [-4.35433, -2.90931, 1.67463, 0.14073],
                    [-4.47182, -3.10839, 1.67463, 0.13804],
                    [-4.55583, -3.32036, 1.67463, 0.13615],
                    [-4.60319, -3.54161, 1.67463, 0.13511],
                    [-4.61034, -3.76744, 1.69781, 0.13308],
                    [-4.57797, -3.99216, 1.71074, 0.13272],
                    [-4.50821, -4.21089, 1.71074, 0.1342],
                    [-4.40027, -4.41828, 1.71074, 0.13666],
                    [-4.25336, -4.60766, 1.80255, 0.13297],
                    [-4.0722, -4.77468, 2.00785, 0.12272],
                    [-3.86422, -4.91915, 2.13353, 0.11869],
                    [-3.63312, -5.03996, 2.28296, 0.11423],
                    [-3.38246, -5.13685, 2.46406, 0.10906],
                    [-3.11571, -5.21042, 2.69587, 0.10264],
                    [-2.83623, -5.26225, 3.00782, 0.0945],
                    [-2.54725, -5.29499, 3.45878, 0.08408],
                    [-2.25171, -5.31226, 3.75, 0.07895],
                    [-1.95217, -5.3185, 3.75, 0.07989],
                    [-1.65075, -5.31872, 3.75, 0.08038],
                    [-1.34911, -5.31817, 3.75, 0.08044],
                    [-1.04746, -5.31761, 3.75, 0.08044],
                    [-0.74582, -5.31704, 3.75, 0.08044],
                    [-0.44418, -5.31646, 3.75, 0.08044],
                    [-0.14253, -5.31589, 3.75, 0.08044],
                    [0.15911, -5.31532, 3.75, 0.08044],
                    [0.46076, -5.31475, 3.75, 0.08044],
                    [0.7624, -5.31418, 3.6645, 0.08232],
                    [1.06405, -5.31361, 3.31155, 0.09109],
                    [1.3651, -5.31134, 3.04109, 0.099],
                    [1.66434, -5.30413, 2.8237, 0.106],
                    [1.96039, -5.28888, 2.6433, 0.11215],
                    [2.25178, -5.26278, 2.48902, 0.11754],
                    [2.53695, -5.22338, 2.35465, 0.12226],
                    [2.81429, -5.16859, 2.23483, 0.1265],
                    [3.08218, -5.09667, 2.1264, 0.13044],
                    [3.33891, -5.00619, 2.1264, 0.12801],
                    [3.58269, -4.89597, 2.1264, 0.12582],
                    [3.81155, -4.76499, 2.1264, 0.12401],
                    [4.0232, -4.61231, 2.1264, 0.12273],
                    [4.21488, -4.43704, 2.1264, 0.12215],
                    [4.38302, -4.23837, 2.30461, 0.11294],
                    [4.52963, -4.02084, 2.42147, 0.10833],
                    [4.65535, -3.78689, 2.55774, 0.10384],
                    [4.76096, -3.53862, 2.71112, 0.09951],
                    [4.84733, -3.27794, 2.89276, 0.09493],
                    [4.91558, -3.00662, 3.11478, 0.08982],
                    [4.96709, -2.72638, 3.39407, 0.08395],
                    [5.00362, -2.43892, 3.75, 0.07727],
                    [5.0273, -2.14585, 3.75, 0.07841],
                    [5.04064, -1.8487, 3.75, 0.07932],
                    [5.04646, -1.5489, 3.75, 0.07996],
                    [5.04781, -1.24766, 3.75, 0.08033],
                    [5.04781, -0.94602, 3.75, 0.08044],
                    [5.04781, -0.64437, 3.75, 0.08044],
                    [5.04779, -0.34273, 3.75, 0.08044],
                    [5.04777, -0.04108, 3.75, 0.08044],
                    [5.04775, 0.21504, 3.75, 0.0683],
                    [5.04773, 0.45271, 3.75, 0.06338]]
    ################## INPUT PARAMETERS ###################

    # Read all input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    x = params['x']
    y = params['y']
    distance_from_center = params['distance_from_center']
    is_left_of_center = params['is_left_of_center']
    heading = params['heading']
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    steering_angle = params['steering_angle']
    track_width = params['track_width']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    is_offtrack = params['is_offtrack']
    is_crashed = params['is_crashed']

    ############### OPTIMAL X,Y,SPEED,TIME ################

    # Get closest indexes for racing line (and distances to all points on racing line)
    closest_index, second_closest_index = closest_2_racing_points_index(
        racing_track, [x, y])

    # Get optimal [x, y, speed, time] for closest and second closest index
    optimals = racing_track[closest_index]
    optimals_second = racing_track[second_closest_index]

    # Save first racingpoint of episode for later
    if steps == 1:
        first_racingpoint_index = closest_index
    else:
        first_racingpoint_index = 0

    ################ REWARD AND PUNISHMENT ################

    ## Define the default reward ##
    reward = 1

    ## Reward if car goes close to optimal racing line ##
    DISTANCE_MULTIPLE = 1
    dist = dist_to_racing_line(optimals[0:2], optimals_second[0:2], [x, y])
    distance_reward = max(1e-3, 1 - (dist/(track_width*0.6)))
    reward += distance_reward * DISTANCE_MULTIPLE

    ## Reward if speed is close to optimal speed ##
    SPEED_DIFF_NO_REWARD = 1
    SPEED_MULTIPLE = 2
    speed_diff = abs(optimals[2]-speed)
    if speed_diff <= SPEED_DIFF_NO_REWARD:
        speed_reward = (1 - (speed_diff/(SPEED_DIFF_NO_REWARD))**2)**2
    else:
        speed_reward = 0
    reward += speed_reward * SPEED_MULTIPLE

    # Reward if less steps
    REWARD_PER_STEP_FOR_FASTEST_TIME = 1
    STANDARD_TIME = 21
    FASTEST_TIME = 18
    times_list = [row[3] for row in racing_track]
    projected_time = projected_time(first_racingpoint_index, closest_index, steps, times_list)
    try:
        steps_prediction = projected_time * 15 + 1
        reward_prediction = max(1e-3, (-REWARD_PER_STEP_FOR_FASTEST_TIME*(FASTEST_TIME) /
                                       (STANDARD_TIME-FASTEST_TIME))*(steps_prediction-(STANDARD_TIME*15+1)))
        steps_reward = min(REWARD_PER_STEP_FOR_FASTEST_TIME, reward_prediction / steps_prediction)
    except:
        steps_reward = 0
    reward += steps_reward

    # Zero reward if obviously wrong direction (e.g. spin)
    direction_diff = racing_direction_diff(
        optimals[0:2], optimals_second[0:2], [x, y], heading)
    if direction_diff > 30:
        reward = 1e-3

    # Zero reward of obviously too slow
    speed_diff_zero = optimals[2]-speed
    if speed_diff_zero > 0.5:
        reward = 1e-3

    ## Incentive for finishing the lap in less steps ##
    REWARD_FOR_FASTEST_TIME = 1000 # should be adapted to track length and other rewards
    STANDARD_TIME = 22  # seconds (time that is easily done by model)
    FASTEST_TIME = 18  # seconds (best time of 1st place on the track)
    if progress == 100:
        finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                                   (15*(STANDARD_TIME-FASTEST_TIME)))*(steps-STANDARD_TIME*15))
    else:
        finish_reward = 0
    reward += finish_reward

    ## Zero reward if off track ##
    if not all_wheels_on_track:
        reward = 1e-3
    else:
        reward *= 1.02

    # Always return a float value
    return float(reward)