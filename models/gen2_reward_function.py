import math
import numpy as np


def reward_function(params):
    # input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    x = params['x']
    y = params['y']
    distance_from_center = params['distance_from_center']
    is_left_of_center = params['is_left_of_center']
    heading = params['heading']  # yaw
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    steering_angle = params['steering_angle']
    track_width = params['track_width']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    is_offtrack = params['is_offtrack']
    is_crashed = params['is_crashed']

    reward = 1.0
    MIN_SPEED = 0.5
    MAX_SPEED = 2.5
    # OPTIMAL_SPEED = abs((MIN_SPEED + MAX_SPEED) / 2)
    STEP_INTERVAL = 4  # steps to complete before evaluation
    HEADING_THRESHOLD = 12.5  # yaw, agent heading
    LOOK_AHEAD = 5  # qty upcoming points to consider for curvature
    MAX_DISTANCE = 0.06  # acceptable distance from optimized race line
    LINEAR_THRESHOLD = 0.30  # acceptable diff to satisfy linear regression
    STEERING_ANGLE_THRESHOLD = 10.0  # acceptable steering angle cap
    CURRENT_INDEX = params['closest_waypoints'][1]

    # Optimized race line coordinates
    optimized_race_line = np.array([[8.48204371, 3.06614279],
                                    [8.30473316, 3.10632633],
                                    [8.12152853, 3.138318  ],
                                    [7.93306693, 3.16284207],
                                    [7.74013035, 3.18078412],
                                    [7.54378164, 3.19318756],
                                    [7.34580453, 3.20125366],
                                    [7.1481691 , 3.20634061],
                                    [6.95096435, 3.2097044 ],
                                    [6.75393389, 3.21157939],
                                    [6.55694066, 3.21218784],
                                    [6.35990892, 3.21174206],
                                    [6.16280427, 3.21045081],
                                    [5.96562187, 3.20852614],
                                    [5.76837641, 3.20617632],
                                    [5.57109567, 3.20361328],
                                    [5.37377901, 3.20085269],
                                    [5.1750152 , 3.1979687 ],
                                    [4.97478628, 3.19527119],
                                    [4.77399871, 3.19279315],
                                    [4.57301635, 3.1905422 ],
                                    [4.37195165, 3.18854667],
                                    [4.17085227, 3.18682057],
                                    [3.96973655, 3.18541639],
                                    [3.76861304, 3.18443716],
                                    [3.56748821, 3.18396707],
                                    [3.3663686 , 3.18407862],
                                    [3.16526208, 3.18483626],
                                    [2.96417904, 3.18630209],
                                    [2.7631332 , 3.1885326 ],
                                    [2.56214266, 3.19158276],
                                    [2.3612315 , 3.19550818],
                                    [2.16043335, 3.20036656],
                                    [1.96842774, 3.20174402],
                                    [1.78228065, 3.19668642],
                                    [1.60313888, 3.18271433],
                                    [1.43177104, 3.1579623 ],
                                    [1.26883521, 3.12114713],
                                    [1.11506182, 3.07143575],
                                    [0.97132761, 3.00837937],
                                    [0.83875076, 2.93181054],
                                    [0.71852265, 2.84203494],
                                    [0.61206234, 2.73961967],
                                    [0.52092558, 2.62549131],
                                    [0.44645243, 2.50111123],
                                    [0.38995649, 2.36821829],
                                    [0.35259963, 2.22885138],
                                    [0.33496413, 2.08533945],
                                    [0.33733199, 1.94004467],
                                    [0.35959379, 1.79528178],
                                    [0.40136167, 1.65321837],
                                    [0.46199914, 1.51580747],
                                    [0.54068807, 1.38473914],
                                    [0.63643434, 1.26138127],
                                    [0.74833105, 1.14687972],
                                    [0.87538574, 1.04208185],
                                    [1.01661737, 0.94758647],
                                    [1.17102536, 0.8637386 ],
                                    [1.33765968, 0.79072819],
                                    [1.51532542, 0.72841152],
                                    [1.70240225, 0.67638092],
                                    [1.89602815, 0.63396168],
                                    [2.09232238, 0.60000575],
                                    [2.28927288, 0.57253176],
                                    [2.48649131, 0.54988062],
                                    [2.68387818, 0.53154563],
                                    [2.8813297 , 0.51733226],
                                    [3.07854002, 0.50710304],
                                    [3.27471966, 0.50073076],
                                    [3.46916515, 0.49800615],
                                    [3.66170619, 0.49867877],
                                    [3.85247575, 0.50245019],
                                    [4.04168169, 0.50904647],
                                    [4.22954815, 0.51818392],
                                    [4.41628529, 0.52957623],
                                    [4.60207599, 0.54295066],
                                    [4.787087  , 0.55802873],
                                    [4.97145812, 0.57455759],
                                    [5.15532035, 0.592278  ],
                                    [5.33878972, 0.61094552],
                                    [5.51862434, 0.6277076 ],
                                    [5.69789385, 0.64224246],
                                    [5.87656601, 0.65396409],
                                    [6.05463187, 0.66235488],
                                    [6.23210071, 0.667018  ],
                                    [6.40898306, 0.66763414],
                                    [6.58529073, 0.66398535],
                                    [6.76102769, 0.6558908 ],
                                    [6.93620425, 0.64330685],
                                    [7.11082724, 0.62623465],
                                    [7.28490667, 0.60475257],
                                    [7.4584498 , 0.57895059],
                                    [7.63146338, 0.54893764],
                                    [7.80395114, 0.51481639],
                                    [7.97591368, 0.47668334],
                                    [8.14734316, 0.43460747],
                                    [8.29848554, 0.40184233],
                                    [8.44753964, 0.3759125 ],
                                    [8.59370787, 0.3588781 ],
                                    [8.73635121, 0.35203117],
                                    [8.87490353, 0.35624482],
                                    [9.00890738, 0.37194702],
                                    [9.13786201, 0.39954513],
                                    [9.26129088, 0.43926093],
                                    [9.37856415, 0.49150305],
                                    [9.48885165, 0.55686052],
                                    [9.59098425, 0.63620606],
                                    [9.68425856, 0.72935917],
                                    [9.76762368, 0.83652549],
                                    [9.83959462, 0.9581973 ],
                                    [9.89809452, 1.09505228],
                                    [9.94033101, 1.24760231],
                                    [9.96266221, 1.41542994],
                                    [9.96101562, 1.59522875],
                                    [9.93329302, 1.77804211],
                                    [9.88178989, 1.95255814],
                                    [9.81110207, 2.11350879],
                                    [9.72482424, 2.26041483],
                                    [9.62534401, 2.39416202],
                                    [9.51428573, 2.515677  ],
                                    [9.3928048 , 2.62567869],
                                    [9.26172313, 2.72465968],
                                    [9.12169459, 2.81300869],
                                    [8.97320825, 2.89098047],
                                    [8.81677623, 2.95891003],
                                    [8.65287955, 3.01714506],
                                    [8.48204371, 3.06614279]])

    def calc_centerline_heading_diff(waypoints, closest_waypoints, agent_heading):
        # return difference in heading value, between agent and centerline
        def calc_heading(point1, point2, in_degrees: bool = True):
            # return a heading in degrees (easier conditions) or radians (easier math)
            delta_y = point2[1] - point1[1]
            delta_x = point2[0] - point1[0]
            heading_rad = math.atan2(delta_y, delta_x)
            return math.degrees(heading_rad) if in_degrees else heading_rad

        agent_heading_rad = math.radians(agent_heading)
        if closest_waypoints[1] < len(waypoints) - 1:
            center_heading_rad = calc_heading(waypoints[closest_waypoints[0]], waypoints[closest_waypoints[1]], False)
        else:
            center_heading_rad = calc_heading(waypoints[closest_waypoints[0]], waypoints[closest_waypoints[0] - 1],
                                              False)

        center_agent_heading_diff = abs(agent_heading_rad - center_heading_rad)
        return center_agent_heading_diff

    def get_intermediate_rewards(speed, progress, steps, steering_angle, heading, optimized_line,
                                 optimized_waypoints):
        def calc_steering_ir(steering_angle, speed):
            # reward smaller steering angles and throttle limits
            _steering_reward = 1
            # diff between current/max angle, as float
            _remaining_cap = 1 - abs(steering_angle) / STEERING_ANGLE_THRESHOLD
            if abs(steering_angle) < STEERING_ANGLE_THRESHOLD:
                _steering_reward += 1 * _remaining_cap
                # penalize excessive speed when high steering angle
                if speed > (MAX_SPEED * _remaining_cap):
                    _steering_reward *= _remaining_cap
                else:
                    _steering_reward += _remaining_cap
            else:  # outside tolerance, penalize base and speed
                _steering_reward *= 0.50
                if speed > (MAX_SPEED * _remaining_cap):
                    _steering_reward *= _remaining_cap
            return _steering_reward

        def calc_heading_ir(heading, speed, optimized_race_line, closest_optimized_waypoints):
            _heading_reward = 1
            yaw_diff = calc_centerline_heading_diff(optimized_race_line, closest_optimized_waypoints, heading)
            if abs(yaw_diff) < HEADING_THRESHOLD:
                _heading_reward += 1
            elif abs(yaw_diff) < (HEADING_THRESHOLD * 1.10):  # 110% threshold
                _heading_reward += 1 * (HEADING_THRESHOLD * 0.90)  # 10% reduction
                if speed > (MAX_SPEED * 0.50):
                    _heading_reward *= 0.50
            else:
                if speed > (MAX_SPEED * 0.50):
                    _heading_reward *= 0.20
                else:
                    _heading_reward *= 0.25
            return _heading_reward

        def calc_step_ir(progress, steps):
            # reward intermediate and milestone progress
            _step_ir = 1
            if steps % STEP_INTERVAL == 0:
                _step_ir += 1 * abs(progress / steps)
            if round(progress) in [10, 25, 50, 75, 100]:
                _step_ir += abs(progress * 0.10)
            return _step_ir

        speed_ir = 1 if MIN_SPEED < speed < MAX_SPEED else 0
        steering_ir = calc_steering_ir(steering_angle, speed)
        heading_ir = calc_heading_ir(heading, speed, optimized_line, optimized_waypoints)
        step_ir = calc_step_ir(progress, steps)
        return speed_ir + steering_ir + heading_ir + step_ir

    def get_speed_angle_reward(curve, speed):
        # return reward based on speed and angle ratio
        line_ir = 0
        if curve >= 0:
            if curve > 0.30:  # penalize high angle+high speed
                line_ir += 1 if speed < MAX_SPEED * (1 - curve) else -1
            else:  # penalize low angle+low speed
                line_ir += 1 if speed > MAX_SPEED * curve else -1
        else:
            line_ir = 1e-3
        return line_ir

    def calc_curvature(optimized_race_line, current_index, num_points):
        def curvature(x1, y1, x2, y2, x3, y3):
            numerator = 2 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            denominator = np.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2) * ((x3 - x2) ** 2 + (y3 - y2) ** 2) * (
                        (x3 - x1) ** 2 + (y3 - y1) ** 2))
            if denominator == 0:
                return 0  # To avoid division by zero
            return numerator / denominator

        curvatures = []

        for i in range(current_index, current_index + num_points - 2):
            i1 = i % len(optimized_race_line)
            i2 = (i + 1) % len(optimized_race_line)
            i3 = (i + 2) % len(optimized_race_line)

            x1, y1 = optimized_race_line[i1]
            x2, y2 = optimized_race_line[i2]
            x3, y3 = optimized_race_line[i3]

            curv = curvature(x1, y1, x2, y2, x3, y3)
            curvatures.append(curv)

        return curvatures

    def calc_normalized_curve(optimized_race_line, look_ahead):
        curvatures = calc_curvature(optimized_race_line, CURRENT_INDEX, num_points=look_ahead)
        average_curve = np.mean(curvatures)
        return average_curve / (average_curve + 1)

    # Calculate normalized curve
    car_position = np.array([x, y])
    distances = np.linalg.norm(optimized_race_line - car_position, axis=1)
    min_distance = np.min(distances)
    normalized_curve = calc_normalized_curve(optimized_race_line, LOOK_AHEAD)
    # Find nearest Optimized waypoints/indexes
    closest_opt_index = np.argmin(distances)
    previous_opt_waypoint = closest_opt_index - 1 if closest_opt_index > 0 else len(optimized_race_line) - 1
    next_opt_waypoint = closest_opt_index + 1 if closest_opt_index < len(optimized_race_line) - 1 else 0
    closest_optimized_waypoints = [previous_opt_waypoint, next_opt_waypoint]

    # Apply intermediate, base reward values
    reward += get_intermediate_rewards(speed, progress, steps, steering_angle, heading, optimized_race_line,
                                       closest_optimized_waypoints)

    # Apply reward for speed:angle ratio
    reward += get_speed_angle_reward(normalized_curve, speed)

    # Reward for staying close to the optimized race line
    obedient_reward = 0
    if min_distance < MAX_DISTANCE:
        obedient_reward = 4.00 * (MAX_DISTANCE - min_distance) / MAX_DISTANCE
        if min_distance < (MAX_DISTANCE / 2):
            obedient_reward *= 1.5  # bonus multiplier for being closer 
    else:
        reward = 1e-3  # Minimum reward if too far from the race line
    reward += obedient_reward

    if is_offtrack or is_crashed or not all_wheels_on_track:
        reward = 1e-3

    return float(reward)
