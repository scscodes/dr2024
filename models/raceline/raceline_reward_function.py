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
    racing_track = [[5.04772, 0.65863, 4.0, 0.05148],
                    [5.04771, 0.84177, 4.0, 0.04579],
                    [5.04769, 1.00937, 4.0, 0.0419],
                    [5.04705, 1.19409, 3.73522, 0.04945],
                    [5.04292, 1.4719, 3.43443, 0.0809],
                    [5.03191, 1.76493, 3.19246, 0.09185],
                    [5.01149, 2.05701, 2.99012, 0.09792],
                    [4.97951, 2.34448, 2.81496, 0.10275],
                    [4.93409, 2.62564, 2.66446, 0.10689],
                    [4.87363, 2.89905, 2.53173, 0.1106],
                    [4.79678, 3.1633, 2.38077, 0.11559],
                    [4.70238, 3.41696, 2.38077, 0.11369],
                    [4.58936, 3.65848, 2.38077, 0.112],
                    [4.45681, 3.88613, 2.38077, 0.11065],
                    [4.30384, 4.09789, 2.38077, 0.10973],
                    [4.12913, 4.29078, 2.47141, 0.1053],
                    [3.9353, 4.46508, 2.57139, 0.10137],
                    [3.72455, 4.62117, 2.68015, 0.09785],
                    [3.49873, 4.75944, 2.79962, 0.09458],
                    [3.25947, 4.88032, 2.9319, 0.09143],
                    [3.00827, 4.98429, 3.0827, 0.08819],
                    [2.74654, 5.07196, 3.25674, 0.08475],
                    [2.47564, 5.14411, 3.46207, 0.08098],
                    [2.1969, 5.20169, 3.70848, 0.07675],
                    [1.9116, 5.24588, 4.0, 0.07217],
                    [1.62101, 5.27811, 4.0, 0.07309],
                    [1.32629, 5.3, 4.0, 0.07388],
                    [1.02855, 5.3134, 4.0, 0.07451],
                    [0.72874, 5.32026, 4.0, 0.07497],
                    [0.42768, 5.32264, 4.0, 0.07527],
                    [0.12603, 5.32269, 3.7938, 0.07951],
                    [-0.17498, 5.31829, 3.49552, 0.08612],
                    [-0.4748, 5.30687, 3.25747, 0.09211],
                    [-0.77264, 5.28589, 3.05862, 0.09762],
                    [-1.06746, 5.253, 2.89177, 0.10258],
                    [-1.35791, 5.20601, 2.74952, 0.10701],
                    [-1.64229, 5.14305, 2.62358, 0.11102],
                    [-1.91861, 5.06258, 2.50963, 0.11468],
                    [-2.18456, 4.96347, 2.40782, 0.11787],
                    [-2.43762, 4.84509, 2.30956, 0.12096],
                    [-2.67512, 4.70736, 2.22083, 0.12362],
                    [-2.89435, 4.55073, 2.13853, 0.12599],
                    [-3.09257, 4.37615, 2.06058, 0.12819],
                    [-3.26719, 4.18509, 1.98624, 0.13031],
                    [-3.41563, 3.97939, 1.91592, 0.1324],
                    [-3.53552, 3.76136, 1.84852, 0.1346],
                    [-3.62467, 3.53369, 1.78586, 0.13691],
                    [-3.68099, 3.29944, 1.72494, 0.13967],
                    [-3.70243, 3.0621, 1.72494, 0.13816],
                    [-3.68694, 2.8257, 1.72494, 0.13734],
                    [-3.63244, 2.59505, 1.72494, 0.13739],
                    [-3.53695, 2.37614, 1.72494, 0.13846],
                    [-3.39892, 2.17683, 1.98327, 0.12224],
                    [-3.22996, 1.99728, 2.10845, 0.11693],
                    [-3.03435, 1.83871, 2.2532, 0.11176],
                    [-2.81588, 1.70156, 2.43044, 0.10613],
                    [-2.57813, 1.58541, 2.65415, 0.0997],
                    [-2.32446, 1.48903, 2.94901, 0.09202],
                    [-2.05808, 1.41032, 3.36728, 0.08249],
                    [-1.78207, 1.34629, 3.43684, 0.08244],
                    [-1.49937, 1.29325, 2.94703, 0.0976],
                    [-1.21286, 1.24681, 2.6185, 0.11085],
                    [-0.93206, 1.19646, 2.3778, 0.11998],
                    [-0.6576, 1.13673, 2.19032, 0.12824],
                    [-0.39306, 1.0633, 2.03933, 0.13462],
                    [-0.14198, 0.97291, 1.91394, 0.13943],
                    [0.09217, 0.86338, 1.80454, 0.14325],
                    [0.30597, 0.73362, 1.71034, 0.14623],
                    [0.49596, 0.58349, 1.62597, 0.14893],
                    [0.65864, 0.4138, 1.55, 0.15166],
                    [0.79031, 0.22623, 1.55, 0.14785],
                    [0.88696, 0.02339, 1.55, 0.14496],
                    [0.94425, -0.19092, 1.55, 0.14312],
                    [0.95749, -0.41126, 1.55, 0.14241],
                    [0.92198, -0.62962, 1.72679, 0.12812],
                    [0.84755, -0.83992, 1.8048, 0.1236],
                    [0.7377, -1.03812, 1.88899, 0.11996],
                    [0.59559, -1.22111, 1.98279, 0.11685],
                    [0.42414, -1.38635, 2.09045, 0.11391],
                    [0.22621, -1.53189, 2.21473, 0.11093],
                    [0.00469, -1.6563, 2.36208, 0.10756],
                    [-0.23744, -1.7588, 2.5427, 0.10341],
                    [-0.49702, -1.83943, 2.77223, 0.09805],
                    [-0.77079, -1.8992, 3.07623, 0.09109],
                    [-1.05548, -1.94017, 3.48482, 0.08254],
                    [-1.34806, -1.96525, 3.35516, 0.08752],
                    [-1.64577, -1.97822, 3.02296, 0.09858],
                    [-1.94628, -1.98353, 2.77035, 0.10849],
                    [-2.2426, -1.99903, 2.57167, 0.11538],
                    [-2.53217, -2.02826, 2.40736, 0.1209],
                    [-2.81236, -2.07401, 2.26922, 0.12511],
                    [-3.08045, -2.13831, 2.14854, 0.12832],
                    [-3.33379, -2.22233, 2.0445, 0.13055],
                    [-3.56986, -2.32648, 1.95079, 0.13226],
                    [-3.78624, -2.45054, 1.86634, 0.13364],
                    [-3.98062, -2.59378, 1.7859, 0.1352],
                    [-4.15068, -2.75504, 1.71196, 0.13689],
                    [-4.29409, -2.93273, 1.69467, 0.13475],
                    [-4.40836, -3.12491, 1.69467, 0.13193],
                    [-4.49073, -3.32914, 1.69467, 0.12994],
                    [-4.538, -3.54231, 1.69467, 0.12884],
                    [-4.54657, -3.7602, 1.69467, 0.12868],
                    [-4.51516, -3.9773, 1.75135, 0.12525],
                    [-4.44776, -4.18889, 1.75135, 0.12679],
                    [-4.34324, -4.38989, 1.75135, 0.12936],
                    [-4.20057, -4.5739, 1.96932, 0.11823],
                    [-4.02798, -4.73963, 2.07969, 0.11506],
                    [-3.82867, -4.88501, 2.20611, 0.11182],
                    [-3.60581, -5.00863, 2.35447, 0.10824],
                    [-3.36256, -5.10981, 2.5324, 0.10403],
                    [-3.10211, -5.18866, 2.75643, 0.09872],
                    [-2.82769, -5.24619, 3.05205, 0.09187],
                    [-2.54248, -5.28448, 3.46755, 0.08299],
                    [-2.24951, -5.30662, 4.0, 0.07345],
                    [-1.9515, -5.31651, 4.0, 0.07454],
                    [-1.65075, -5.31872, 4.0, 0.07519],
                    [-1.34911, -5.31817, 4.0, 0.07541],
                    [-1.04746, -5.31761, 4.0, 0.07541],
                    [-0.74582, -5.31704, 4.0, 0.07541],
                    [-0.44418, -5.31646, 4.0, 0.07541],
                    [-0.14253, -5.31589, 4.0, 0.07541],
                    [0.15911, -5.31532, 4.0, 0.07541],
                    [0.46076, -5.31475, 4.0, 0.07541],
                    [0.7624, -5.31418, 4.0, 0.07541],
                    [1.06374, -5.31279, 3.72772, 0.08084],
                    [1.36362, -5.30766, 3.41046, 0.08794],
                    [1.66079, -5.296, 3.16032, 0.09411],
                    [1.95392, -5.2752, 2.95586, 0.09942],
                    [2.24158, -5.24297, 2.78595, 0.1039],
                    [2.52237, -5.19733, 2.62945, 0.10819],
                    [2.79484, -5.1366, 2.49495, 0.11189],
                    [3.05756, -5.05942, 2.3757, 0.11526],
                    [3.30906, -4.96467, 2.26641, 0.11858],
                    [3.54785, -4.85147, 2.26641, 0.1166],
                    [3.77217, -4.71888, 2.26641, 0.11497],
                    [3.98005, -4.56609, 2.26641, 0.11383],
                    [4.16912, -4.3923, 2.26641, 0.11331],
                    [4.33632, -4.1967, 2.4471, 0.10515],
                    [4.48352, -3.98327, 2.56384, 0.10113],
                    [4.61127, -3.75406, 2.69465, 0.09738],
                    [4.72018, -3.51087, 2.84758, 0.09358],
                    [4.81099, -3.25535, 3.02307, 0.0897],
                    [4.88458, -2.98904, 3.2345, 0.08542],
                    [4.94205, -2.71345, 3.49295, 0.08059],
                    [4.98479, -2.43008, 3.82056, 0.07501],
                    [5.0145, -2.1404, 4.0, 0.0728],
                    [5.03322, -1.84584, 4.0, 0.07379],
                    [5.04327, -1.54778, 4.0, 0.07456],
                    [5.04723, -1.24747, 4.0, 0.07508],
                    [5.04781, -0.94602, 4.0, 0.07536],
                    [5.04781, -0.64437, 4.0, 0.07541],
                    [5.04779, -0.34273, 4.0, 0.07541],
                    [5.04777, -0.04108, 4.0, 0.07541],
                    [5.04775, 0.21504, 4.0, 0.06403],
                    [5.04773, 0.45271, 4.0, 0.05942]]
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

    ## Define the default reward
    reward = 1

    ## Reward if car goes close to optimal racing line
    DISTANCE_MULTIPLE = 2.1
    dist = dist_to_racing_line(optimals[0:2], optimals_second[0:2], [x, y])
    distance_reward = max(1e-3, 1 - (dist/(track_width*0.75)))
    reward += distance_reward * DISTANCE_MULTIPLE

    ## Reward if speed is close to optimal speed
    SPEED_DIFF_NO_REWARD = 1
    SPEED_MULTIPLE = 2
    speed_diff = abs(optimals[2]-speed)
    if speed_diff <= SPEED_DIFF_NO_REWARD:
        speed_reward = (1 - (speed_diff/(SPEED_DIFF_NO_REWARD))**2)**2
    else:
        speed_reward = 0
    reward += speed_reward * SPEED_MULTIPLE

    # Reward if less steps
    REWARD_PER_STEP_FOR_FASTEST_TIME = 2
    STANDARD_TIME = 20
    FASTEST_TIME = 17.5
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
    if direction_diff > 25:
        reward = 1e-3

    # Zero reward of obviously too slow
    speed_diff_zero = optimals[2]-speed
    if speed_diff_zero > 0.5:
        reward = 1e-3

    ## Incentive for finishing the lap in less steps
    REWARD_FOR_FASTEST_TIME = 750
    STANDARD_TIME = 20
    FASTEST_TIME = 17.5
    if progress == 100:
        finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                                   (15*(STANDARD_TIME-FASTEST_TIME)))*(steps-STANDARD_TIME*15))
    else:
        finish_reward = 0
    reward += finish_reward

    ## Penalize off track
    if not all_wheels_on_track:
        reward = 1e-3

    # Always return a float value
    return float(reward)
