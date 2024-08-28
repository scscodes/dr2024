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
    racing_track = [[1.55016, -1.39973, 1.05355, 0.47455],
                    [2.04229, -1.35448, 0.98352, 0.50249],
                    [2.52441, -1.28667, 0.98352, 0.49502],
                    [2.9935, -1.19086, 0.98352, 0.48679],
                    [3.44647, -1.06234, 0.98352, 0.47874],
                    [3.88003, -0.89687, 0.98352, 0.47184],
                    [4.2903, -0.6903, 0.98352, 0.46704],
                    [4.67248, -0.43856, 1.02636, 0.44588],
                    [5.02732, -0.14699, 1.01911, 0.45066],
                    [5.35259, 0.18376, 0.81976, 0.56589],
                    [5.64525, 0.55329, 0.69302, 0.68019],
                    [5.90127, 0.96105, 0.61031, 0.78888],
                    [6.1151, 1.40567, 0.5511, 0.89525],
                    [6.27964, 1.8824, 0.5049, 0.99887],
                    [6.38529, 2.36769, 0.5049, 0.98368],
                    [6.40367, 2.82709, 0.5, 0.91953],
                    [6.32783, 3.23579, 0.5, 0.83135],
                    [6.16614, 3.57542, 0.5, 0.75232],
                    [5.93285, 3.83257, 0.5, 0.69441],
                    [5.64366, 3.99325, 0.5, 0.66166],
                    [5.32142, 4.06089, 0.5, 0.65853],
                    [4.98217, 4.01893, 0.53418, 0.63991],
                    [4.64865, 3.86678, 0.58764, 0.62383],
                    [4.34213, 3.60818, 0.65954, 0.60805],
                    [4.08076, 3.25138, 0.67654, 0.65375],
                    [3.88092, 2.82579, 0.67654, 0.69497],
                    [3.62391, 2.4815, 0.67654, 0.63504],
                    [3.32282, 2.21154, 0.67654, 0.59774],
                    [2.98553, 2.01366, 0.67654, 0.57801],
                    [2.61712, 1.8908, 0.74053, 0.52444],
                    [2.22707, 1.83304, 0.79224, 0.4977],
                    [1.82237, 1.83663, 0.85597, 0.47282],
                    [1.41386, 1.8965, 0.93168, 0.44314],
                    [0.99829, 1.90842, 1.00866, 0.41217],
                    [0.57969, 1.87829, 0.88914, 0.47201],
                    [0.15933, 1.81208, 0.79174, 0.53748],
                    [-0.2618, 1.71583, 0.79174, 0.54561],
                    [-0.66474, 1.60174, 0.79174, 0.52893],
                    [-1.06586, 1.51539, 0.79174, 0.51823],
                    [-1.46528, 1.46805, 0.79174, 0.50801],
                    [-1.86365, 1.47045, 0.7705, 0.51704],
                    [-2.26119, 1.53566, 0.65019, 0.61959],
                    [-2.65772, 1.63952, 0.65019, 0.63043],
                    [-3.05302, 1.77305, 0.65019, 0.64174],
                    [-3.43599, 1.87763, 0.65019, 0.61058],
                    [-3.82103, 1.93986, 0.65019, 0.59987],
                    [-4.20689, 1.93992, 0.65019, 0.59346],
                    [-4.58839, 1.85249, 0.68674, 0.56993],
                    [-4.95828, 1.68155, 0.72955, 0.55853],
                    [-5.3081, 1.42857, 0.78043, 0.55318],
                    [-5.62719, 1.09632, 0.84207, 0.54705],
                    [-5.90306, 0.69427, 0.91893, 0.53061],
                    [-6.12536, 0.24301, 1.0103, 0.49792],
                    [-6.2934, -0.23388, 0.87529, 0.57767],
                    [-6.4165, -0.72347, 0.78266, 0.64502],
                    [-6.50293, -1.22077, 0.7125, 0.70843],
                    [-6.55815, -1.72325, 0.65757, 0.76874],
                    [-6.57245, -2.221, 0.65757, 0.75728],
                    [-6.53169, -2.68745, 0.65757, 0.71206],
                    [-6.4323, -3.10946, 0.65757, 0.65934],
                    [-6.27497, -3.48264, 0.65757, 0.61589],
                    [-6.05983, -3.8039, 0.65757, 0.58799],
                    [-5.78398, -4.06763, 0.66574, 0.57325],
                    [-5.44897, -4.27245, 0.95119, 0.41281],
                    [-5.0828, -4.44521, 1.40597, 0.28797],
                    [-4.70139, -4.60211, 1.49833, 0.27526],
                    [-4.70139, -4.60211, 1.15115, 0.0],
                    [-4.41941, -4.21849, 0.9672, 0.49226],
                    [-4.13756, -3.83474, 0.83585, 0.56964],
                    [-3.85643, -3.45028, 0.83585, 0.56981],
                    [-3.57443, -3.07872, 0.83585, 0.55806],
                    [-3.28108, -2.73005, 0.83585, 0.54514],
                    [-2.96748, -2.4147, 0.83585, 0.53209],
                    [-2.62538, -2.14214, 0.83585, 0.5233],
                    [-2.24594, -1.92516, 0.90389, 0.48357],
                    [-1.83444, -1.75728, 0.99051, 0.44869],
                    [-1.39512, -1.63336, 1.10641, 0.41256],
                    [-0.93215, -1.54804, 1.27451, 0.36937],
                    [-0.45023, -1.4947, 1.5509, 0.31264],
                    [0.04518, -1.46478, 1.38352, 0.35873],
                    [0.54793, -1.4479, 1.24483, 0.4041],
                    [1.05103, -1.42859, 1.13936, 0.44189]]
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
    FASTEST_TIME = 24
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
    FASTEST_TIME = 24  # seconds (best time of 1st place on the track)
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