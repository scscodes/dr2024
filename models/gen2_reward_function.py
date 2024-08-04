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
    MIN_SPEED = 1.0
    MAX_SPEED = 3.0
    TRACK_WIDTH = 1.07
    # TRACK_LENGTH = 46.16
    # OPTIMAL_SPEED = abs((MIN_SPEED + MAX_SPEED) / 2)
    STEP_INTERVAL = 4  # steps to complete before evaluation
    HEADING_THRESHOLD = 12.5  # yaw, agent heading
    LOOK_AHEAD = 5  # qty upcoming points to consider for curvature
    MAX_DISTANCE = 0.06  # acceptable distance from optimized race line
    LINEAR_THRESHOLD = 0.30  # acceptable diff to satisfy linear regression
    STEERING_ANGLE_THRESHOLD = 10.0  # acceptable steering angle cap
    CURRENT_INDEX = params['closest_waypoints'][1]

    # Optimized race line coordinates
    optimized_race_line = np.array([[ 5.04771315,  0.73385354],
                                    [ 5.04770565,  0.86385354],
                                    [ 5.04752488,  1.00379612],
                                    [ 5.04601829,  1.20414345],
                                    [ 5.03994307,  1.47311557],
                                    [ 5.02600116,  1.76191145],
                                    [ 5.00189401,  2.05172213],
                                    [ 4.96582171,  2.33698581],
                                    [ 4.91622468,  2.61553037],
                                    [ 4.85177041,  2.88595332],
                                    [ 4.77133984,  3.14700389],
                                    [ 4.67395059,  3.39739714],
                                    [ 4.55875699,  3.6357692 ],
                                    [ 4.42492216,  3.86054687],
                                    [ 4.27168295,  4.06991523],
                                    [ 4.09820469,  4.26155308],
                                    [ 3.90671749,  4.43574428],
                                    [ 3.69902421,  4.5927325 ],
                                    [ 3.47672521,  4.73282678],
                                    [ 3.24119441,  4.85629389],
                                    [ 2.99367483,  4.96340414],
                                    [ 2.7353805 ,  5.05453931],
                                    [ 2.46749674,  5.1301884 ],
                                    [ 2.19125568,  5.19108411],
                                    [ 1.90793165,  5.23825206],
                                    [ 1.61880839,  5.2730189 ],
                                    [ 1.32512922,  5.29697578],
                                    [ 1.02804748,  5.3119132 ],
                                    [ 0.72859687,  5.31977457],
                                    [ 0.42767964,  5.32264495],
                                    [ 0.1260345 ,  5.32269001],
                                    [-0.17553475,  5.32209107],
                                    [-0.47670109,  5.31815158],
                                    [-0.77709149,  5.30820067],
                                    [-1.07611143,  5.28873055],
                                    [-1.37287151,  5.25587125],
                                    [-1.66603835,  5.20596638],
                                    [-1.95349843,  5.13568575],
                                    [-2.23209949,  5.04239457],
                                    [-2.49770755,  4.92456867],
                                    [-2.74563079,  4.78206781],
                                    [-2.97117118,  4.61611511],
                                    [-3.17031335,  4.42921718],
                                    [-3.33996939,  4.22467043],
                                    [-3.47786187,  4.00612186],
                                    [-3.58267443,  3.77741575],
                                    [-3.65385519,  3.54233331],
                                    [-3.69115933,  3.30446171],
                                    [-3.69454809,  3.06725446],
                                    [-3.66389259,  2.83414651],
                                    [-3.598898  ,  2.60875986],
                                    [-3.49892966,  2.39525451],
                                    [-3.3633466 ,  2.19872181],
                                    [-3.19776381,  2.02088656],
                                    [-3.00611602,  1.86291036],
                                    [-2.79196061,  1.72517113],
                                    [-2.55861142,  1.60730906],
                                    [-2.30924935,  1.50817011],
                                    [-2.04697933,  1.42575478],
                                    [-1.7748333 ,  1.35725225],
                                    [-1.49581087,  1.29907759],
                                    [-1.21285213,  1.24707933],
                                    [-0.93513108,  1.19266539],
                                    [-0.66350655,  1.12988651],
                                    [-0.40197072,  1.05409474],
                                    [-0.15435088,  0.96181218],
                                    [ 0.07571338,  0.85082686],
                                    [ 0.28474953,  0.72013609],
                                    [ 0.46966159,  0.56999618],
                                    [ 0.62724354,  0.40139515],
                                    [ 0.75422073,  0.21612485],
                                    [ 0.84708271,  0.01677691],
                                    [ 0.90195716, -0.19307744],
                                    [ 0.91460233, -0.40845905],
                                    [ 0.8805441 , -0.62211285],
                                    [ 0.8087429 , -0.82836943],
                                    [ 0.70236162, -1.02337644],
                                    [ 0.56427979, -1.20409612],
                                    [ 0.39718567, -1.36805544],
                                    [ 0.20367006, -1.51323014],
                                    [-0.01359087, -1.63811103],
                                    [-0.25174488, -1.74183706],
                                    [-0.50774609, -1.82435222],
                                    [-0.77839586, -1.88656104],
                                    [-1.0605201 , -1.93029929],
                                    [-1.35103   , -1.95843697],
                                    [-1.64709669, -1.97473126],
                                    [-1.94628298, -1.98353201],
                                    [-2.24244866, -1.99956632],
                                    [-2.53263701, -2.02780604],
                                    [-2.81366743, -2.07230487],
                                    [-3.08223703, -2.13593024],
                                    [-3.33511066, -2.22030013],
                                    [-3.56925304, -2.32589768],
                                    [-3.78219789, -2.45201475],
                                    [-3.9715532 , -2.59750105],
                                    [-4.13525744, -2.76061684],
                                    [-4.27157236, -2.93918519],
                                    [-4.3786632 , -3.13085361],
                                    [-4.45462104, -3.33297257],
                                    [-4.49731395, -3.54250904],
                                    [-4.50413531, -3.75581058],
                                    [-4.47213169, -3.96808041],
                                    [-4.40594639, -4.17508205],
                                    [-4.30456706, -4.37236805],
                                    [-4.1668539 , -4.55429217],
                                    [-3.99975286, -4.71932704],
                                    [-3.80600624, -4.8652914 ],
                                    [-3.58840716, -4.99058499],
                                    [-3.34985862, -5.09425553],
                                    [-3.09342189, -5.17615585],
                                    [-2.82222521, -5.23700129],
                                    [-2.5394334 , -5.27851849],
                                    [-2.24811155, -5.30344756],
                                    [-1.95106624, -5.31540212],
                                    [-1.6507535 , -5.31872892],
                                    [-1.34910947, -5.31817913],
                                    [-1.04746497, -5.31761503],
                                    [-0.74582046, -5.31703806],
                                    [-0.44417594, -5.31646395],
                                    [-0.1425315 , -5.31589198],
                                    [ 0.15911295, -5.31531811],
                                    [ 0.46075745, -5.3147459 ],
                                    [ 0.76240206, -5.31417751],
                                    [ 1.06319572, -5.31146533],
                                    [ 1.36196771, -5.30390296],
                                    [ 1.65748335, -5.28900165],
                                    [ 1.94843667, -5.26446612],
                                    [ 2.23350137, -5.22828554],
                                    [ 2.5113553 , -5.17875107],
                                    [ 2.78067877, -5.11442654],
                                    [ 3.04015733, -5.03413435],
                                    [ 3.28845193, -4.93689717],
                                    [ 3.5241493 , -4.82187262],
                                    [ 3.74569189, -4.68829089],
                                    [ 3.95126641, -4.53538429],
                                    [ 4.13870953, -4.36240122],
                                    [ 4.30521967, -4.16853616],
                                    [ 4.4526025 , -3.95745961],
                                    [ 4.581401  , -3.73106306],
                                    [ 4.692197  , -3.49099281],
                                    [ 4.78563276, -3.23873508],
                                    [ 4.86247042, -2.97569186],
                                    [ 4.92363469, -2.70322012],
                                    [ 4.97030934, -2.42268499],
                                    [ 5.00397426, -2.13546103],
                                    [ 5.02640272, -1.84290848],
                                    [ 5.03963323, -1.54634031],
                                    [ 5.04595932, -1.24700049],
                                    [ 5.04784012, -0.94601676],
                                    [ 5.04781485, -0.64437172],
                                    [ 5.047791  , -0.34272665],
                                    [ 5.04776907, -0.04108161],
                                    [ 5.04774594,  0.26056334],
                                    [ 5.04772305,  0.56220838],
                                    [ 5.04771315,  0.73385354]])

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
                _steering_reward += 2 * _remaining_cap
                # penalize excessive speed when high steering angle
                if speed > (MAX_SPEED * (_remaining_cap * 0.50)):
                    _steering_reward *= _remaining_cap
                else:
                    _steering_reward += (_remaining_cap * 1.5)
            else:  # outside tolerance, penalize base and speed
                _steering_reward *= 0.50
                if speed > (MAX_SPEED * _remaining_cap):
                    _steering_reward *= _remaining_cap
            return _steering_reward

        def calc_heading_ir(heading, speed, optimized_race_line, closest_optimized_waypoints):
            _heading_reward = 1
            yaw_diff = calc_centerline_heading_diff(optimized_race_line, closest_optimized_waypoints, heading)
            if abs(yaw_diff) < HEADING_THRESHOLD:
                _heading_reward += 2
            elif abs(yaw_diff) < (HEADING_THRESHOLD * 1.10):  # 110% threshold
                _heading_reward += 1 * (HEADING_THRESHOLD * 0.90)  # 10% reduction
                if speed > (MAX_SPEED * 0.60):
                    _heading_reward *= 0.60
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
                _step_ir += abs(progress * 0.15)
            return _step_ir

        speed_ir = 1 if MIN_SPEED < speed < MAX_SPEED else 0
        steering_ir = calc_steering_ir(steering_angle, speed)
        heading_ir = calc_heading_ir(heading, speed, optimized_line, optimized_waypoints)
        step_ir = calc_step_ir(progress, steps)
        return (speed_ir + steering_ir + heading_ir + step_ir) * 1.05 if all_wheels_on_track else 0.50

    def get_speed_angle_reward(curve, speed):
        # return reward based on speed and angle ratio
        line_ir = 0
        if curve >= 0:
            if curve > 0.30:  # penalize high angle+high speed
                line_ir += 1.5 if speed < MAX_SPEED * (1 - (curve * 0.90)) else 0
            else:  # penalize low angle+low speed
                line_ir += 1.5 if speed > MAX_SPEED * curve else 0
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
            if distance_from_center < (TRACK_WIDTH * 0.50) and all_wheels_on_track:
                obedient_reward += 0.50
    else:
        reward = 1e-3  # Minimum reward if too far from the race line
    reward += obedient_reward

    if is_offtrack or is_crashed:
        reward = 1e-3

    return float(reward)
