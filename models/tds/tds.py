import math
def reward_function(params):

    # Import package (needed for heading)


    ################## HELPER FUNCTIONS ###################

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
    racing_track = [[5.04771, 0.73385, 3.75737, 0.04568],
                    [5.04771, 0.86385, 3.3814, 0.03845],
                    [5.04752, 1.0038, 3.09374, 0.04523],
                    [5.04602, 1.20414, 2.86621, 0.0699],
                    [5.03994, 1.47312, 2.68071, 0.10036],
                    [5.026, 1.76191, 2.52225, 0.11463],
                    [5.00189, 2.05172, 2.38698, 0.12183],
                    [4.96582, 2.33699, 2.26425, 0.12699],
                    [4.91622, 2.61553, 2.15689, 0.13117],
                    [4.85177, 2.88595, 2.05425, 0.13533],
                    [4.77134, 3.147, 2.05425, 0.13297],
                    [4.67395, 3.3974, 2.05425, 0.13079],
                    [4.55876, 3.63577, 2.05425, 0.12888],
                    [4.42492, 3.86055, 2.05425, 0.12735],
                    [4.27168, 4.06992, 2.05425, 0.1263],
                    [4.0982, 4.26155, 2.12502, 0.12164],
                    [3.90672, 4.43574, 2.19869, 0.11774],
                    [3.69902, 4.59273, 2.28327, 0.11402],
                    [3.47673, 4.73283, 2.37206, 0.11077],
                    [3.24119, 4.85629, 2.46778, 0.10776],
                    [2.99367, 4.9634, 2.57803, 0.10462],
                    [2.73538, 5.05454, 2.7027, 0.10134],
                    [2.4675, 5.13019, 2.85504, 0.0975],
                    [2.19126, 5.19108, 3.04541, 0.09289],
                    [1.90793, 5.23825, 3.28703, 0.08738],
                    [1.61881, 5.27302, 3.59833, 0.08093],
                    [1.32513, 5.29698, 4.0, 0.07366],
                    [1.02805, 5.31191, 4.0, 0.07436],
                    [0.7286, 5.31977, 3.98927, 0.07509],
                    [0.42768, 5.32264, 3.34416, 0.08999],
                    [0.12603, 5.32269, 2.94006, 0.1026],
                    [-0.17553, 5.32209, 2.65584, 0.11355],
                    [-0.4767, 5.31815, 2.44709, 0.12308],
                    [-0.77709, 5.3082, 2.28877, 0.13132],
                    [-1.07611, 5.28873, 2.16457, 0.13844],
                    [-1.37287, 5.25587, 2.06185, 0.14481],
                    [-1.66604, 5.20597, 1.98072, 0.15014],
                    [-1.9535, 5.13569, 1.91275, 0.15471],
                    [-2.2321, 5.04239, 1.85063, 0.15876],
                    [-2.49771, 4.92457, 1.80078, 0.16136],
                    [-2.74563, 4.78207, 1.76119, 0.16237],
                    [-2.97117, 4.61612, 1.7246, 0.16237],
                    [-3.17031, 4.42922, 1.69257, 0.16136],
                    [-3.33997, 4.22467, 1.66054, 0.16004],
                    [-3.47786, 4.00612, 1.62834, 0.1587],
                    [-3.58267, 3.77742, 1.59197, 0.15803],
                    [-3.65386, 3.54233, 1.55598, 0.15786],
                    [-3.69116, 3.30446, 1.55598, 0.15474],
                    [-3.69455, 3.06725, 1.55598, 0.15246],
                    [-3.66389, 2.83415, 1.55598, 0.1511],
                    [-3.5989, 2.60876, 1.55598, 0.15075],
                    [-3.49893, 2.39525, 1.55598, 0.15151],
                    [-3.36335, 2.19872, 1.67242, 0.14277],
                    [-3.19776, 2.02089, 1.77737, 0.13671],
                    [-3.00612, 1.86291, 1.90092, 0.13066],
                    [-2.79196, 1.72517, 2.05075, 0.12416],
                    [-2.55861, 1.60731, 2.24061, 0.11668],
                    [-2.30925, 1.50817, 2.49335, 0.10763],
                    [-2.04698, 1.42575, 2.84977, 0.09647],
                    [-1.77483, 1.35725, 2.46588, 0.11381],
                    [-1.49581, 1.29908, 2.17764, 0.13089],
                    [-1.21285, 1.24708, 1.97182, 0.1459],
                    [-0.93513, 1.19267, 1.82467, 0.1551],
                    [-0.66351, 1.12989, 1.70126, 0.16387],
                    [-0.40197, 1.05409, 1.5991, 0.17028],
                    [-0.15435, 0.96181, 1.511, 0.17489],
                    [0.07571, 0.85083, 1.43309, 0.17824],
                    [0.28475, 0.72014, 1.36376, 0.18077],
                    [0.46966, 0.57, 1.3, 0.18322],
                    [0.62724, 0.4014, 1.3, 0.17752],
                    [0.75422, 0.21612, 1.3, 0.17277],
                    [0.84708, 0.01678, 1.3, 0.16917],
                    [0.90196, -0.19308, 1.3, 0.16685],
                    [0.9146, -0.40846, 1.3, 0.16596],
                    [0.88054, -0.62211, 1.44276, 0.14996],
                    [0.80874, -0.82837, 1.50644, 0.14498],
                    [0.70236, -1.02338, 1.57718, 0.14084],
                    [0.56428, -1.2041, 1.65642, 0.1373],
                    [0.39719, -1.36806, 1.74527, 0.13413],
                    [0.20367, -1.51323, 1.849, 0.13084],
                    [-0.01359, -1.63811, 1.97345, 0.12698],
                    [-0.25174, -1.74184, 2.12647, 0.12216],
                    [-0.50775, -1.82435, 2.32089, 0.11589],
                    [-0.7784, -1.88656, 2.57058, 0.10803],
                    [-1.06052, -1.9303, 2.84954, 0.10019],
                    [-1.35103, -1.95844, 2.50513, 0.11651],
                    [-1.6471, -1.97473, 2.26608, 0.13085],
                    [-1.94628, -1.98353, 2.08756, 0.14338],
                    [-2.24245, -1.99957, 1.96087, 0.15126],
                    [-2.53264, -2.02781, 1.84883, 0.1577],
                    [-2.81367, -2.0723, 1.75962, 0.1617],
                    [-3.08224, -2.13593, 1.68939, 0.16337],
                    [-3.33511, -2.2203, 1.62467, 0.16408],
                    [-3.56925, -2.3259, 1.56767, 0.16384],
                    [-3.7822, -2.45201, 1.51541, 0.16332],
                    [-3.97155, -2.5975, 1.46317, 0.1632],
                    [-4.13526, -2.76062, 1.41308, 0.16354],
                    [-4.27157, -2.93919, 1.41308, 0.15898],
                    [-4.37866, -3.13085, 1.41308, 0.15538],
                    [-4.45462, -3.33297, 1.41308, 0.1528],
                    [-4.49731, -3.54251, 1.41308, 0.15133],
                    [-4.50414, -3.75581, 1.41308, 0.15103],
                    [-4.47213, -3.96808, 1.48313, 0.14474],
                    [-4.40595, -4.17508, 1.48313, 0.14653],
                    [-4.30457, -4.37237, 1.48313, 0.14956],
                    [-4.16685, -4.55429, 1.65193, 0.13812],
                    [-3.99975, -4.71933, 1.7399, 0.13498],
                    [-3.80601, -4.86529, 1.84155, 0.13172],
                    [-3.58841, -4.99058, 1.96139, 0.12802],
                    [-3.34986, -5.09426, 2.10782, 0.1234],
                    [-3.09342, -5.17616, 2.28814, 0.11765],
                    [-2.82223, -5.237, 2.5231, 0.11016],
                    [-2.53943, -5.27852, 2.84584, 0.10044],
                    [-2.24811, -5.30345, 3.3244, 0.08795],
                    [-1.95107, -5.3154, 4.0, 0.07432],
                    [-1.65075, -5.31873, 4.0, 0.07508],
                    [-1.34911, -5.31818, 4.0, 0.07541],
                    [-1.04746, -5.31762, 4.0, 0.07541],
                    [-0.74582, -5.31704, 4.0, 0.07541],
                    [-0.44418, -5.31646, 4.0, 0.07541],
                    [-0.14253, -5.31589, 4.0, 0.07541],
                    [0.15911, -5.31532, 3.83895, 0.07857],
                    [0.46076, -5.31475, 3.41138, 0.08842],
                    [0.7624, -5.31418, 3.09905, 0.09733],
                    [1.0632, -5.31147, 2.85536, 0.10535],
                    [1.36197, -5.3039, 2.65891, 0.1124],
                    [1.65748, -5.289, 2.49493, 0.1186],
                    [1.94844, -5.26447, 2.35422, 0.12403],
                    [2.2335, -5.22829, 2.23094, 0.1288],
                    [2.51136, -5.17875, 2.12031, 0.13311],
                    [2.78068, -5.11443, 2.02257, 0.1369],
                    [3.04016, -5.03413, 1.93144, 0.14063],
                    [3.28845, -4.9369, 1.93144, 0.13806],
                    [3.52415, -4.82187, 1.93144, 0.13579],
                    [3.74569, -4.68829, 1.93144, 0.13394],
                    [3.95127, -4.53538, 1.93144, 0.13265],
                    [4.13871, -4.3624, 1.93144, 0.13206],
                    [4.30522, -4.16854, 2.0827, 0.1227],
                    [4.4526, -3.95746, 2.17907, 0.11814],
                    [4.5814, -3.73106, 2.28786, 0.11385],
                    [4.6922, -3.49099, 2.41083, 0.10967],
                    [4.78563, -3.23874, 2.55246, 0.10539],
                    [4.86247, -2.97569, 2.71767, 0.10083],
                    [4.92363, -2.70322, 2.91944, 0.09565],
                    [4.97031, -2.42268, 3.17328, 0.08962],
                    [5.00397, -2.13546, 3.50329, 0.08255],
                    [5.0264, -1.84291, 3.95452, 0.0742],
                    [5.03963, -1.54634, 4.0, 0.07422],
                    [5.04596, -1.247, 4.0, 0.07485],
                    [5.04784, -0.94602, 4.0, 0.07525],
                    [5.04781, -0.64437, 4.0, 0.07541],
                    [5.04779, -0.34273, 4.0, 0.07541],
                    [5.04777, -0.04108, 4.0, 0.07541],
                    [5.04775, 0.26056, 4.0, 0.07541],
                    [5.04772, 0.56221, 4.0, 0.07541]]
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
    distance_reward = max(1e-3, 1 - (dist/(track_width*0.5)))
    reward += distance_reward * DISTANCE_MULTIPLE

    ## Reward if speed is close to optimal speed ##
    SPEED_DIFF_NO_REWARD = 1
    SPEED_MULTIPLE = 2
    speed_diff = abs(optimals[2]-speed)
    if speed_diff <= SPEED_DIFF_NO_REWARD:
        # we use quadratic punishment (not linear) bc we're not as confident with the optimal speed
        # so, we do not punish small deviations from optimal speed
        speed_reward = (1 - (speed_diff/(SPEED_DIFF_NO_REWARD))**2)**2
    else:
        speed_reward = 0
    reward += speed_reward * SPEED_MULTIPLE

    # Reward if less steps
    REWARD_PER_STEP_FOR_FASTEST_TIME = 1
    STANDARD_TIME = 35
    FASTEST_TIME = 22
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
    REWARD_FOR_FASTEST_TIME = 1500 # should be adapted to track length and other rewards
    STANDARD_TIME = 35  # seconds (time that is easily done by model)
    FASTEST_TIME = 22  # seconds (best time of 1st place on the track)
    if progress == 100:
        finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                                   (15*(STANDARD_TIME-FASTEST_TIME)))*(steps-STANDARD_TIME*15))
    else:
        finish_reward = 0
    reward += finish_reward

    ## Zero reward if off track ##
    if not all_wheels_on_track:
        reward = 1e-3

    # Always return a float value
    return float(reward)