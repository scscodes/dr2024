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
    MIN_SPEED = 0.50
    MAX_SPEED = 2.0
    STEP_INTERVAL = 2  # steps to complete before evaluation
    HEADING_THRESHOLD = 30  # yaw, agent heading
    LOOK_AHEAD = 5  # qty upcoming points to consider for curvature
    MAX_DISTANCE = 0.02  # acceptable distance from optimized race line

    # optimized race line (orl) coordinates
    optimized_race_line = np.array([[2.09206132, 1.04584401],
                                    [2.18694718, 1.04403112],
                                    [2.28267068, 1.04979537],
                                    [2.378828  , 1.06279701],
                                    [2.47507081, 1.08266791],
                                    [2.57110689, 1.10894411],
                                    [2.6667081 , 1.14108182],
                                    [2.76170242, 1.1785494 ],
                                    [2.8559754 , 1.22076012],
                                    [2.94946828, 1.26709028],
                                    [3.04468809, 1.3104308 ],
                                    [3.14068735, 1.34998887],
                                    [3.23743124, 1.38548063],
                                    [3.33487225, 1.41665029],
                                    [3.43295249, 1.44325412],
                                    [3.53159675, 1.46512023],
                                    [3.63072016, 1.48211499],
                                    [3.73022996, 1.49413806],
                                    [3.83002426, 1.50117933],
                                    [3.92999979, 1.50322964],
                                    [4.03004902, 1.50034214],
                                    [4.13006289, 1.49263088],
                                    [4.22993284, 1.48027437],
                                    [4.3295521 , 1.46350932],
                                    [4.42881429, 1.4426114 ],
                                    [4.52761612, 1.41792998],
                                    [4.62586036, 1.38988217],
                                    [4.72344692, 1.35889798],
                                    [4.82027857, 1.32546275],
                                    [4.91625823, 1.2900946 ],
                                    [5.01128267, 1.25331856],
                                    [5.10527559, 1.21566183],
                                    [5.19827128, 1.17759499],
                                    [5.28939848, 1.13994964],
                                    [5.38053651, 1.1023717 ],
                                    [5.47168374, 1.06485114],
                                    [5.56283862, 1.02737822],
                                    [5.65399959, 0.98994328],
                                    [5.74516514, 0.95253679],
                                    [5.83633372, 0.91514925],
                                    [5.92750382, 0.87777114],
                                    [5.92750382, 0.87777114],
                                    [5.99593976, 0.94582797],
                                    [6.06382505, 1.01432142],
                                    [6.13066978, 1.08363617],
                                    [6.19601064, 1.15412921],
                                    [6.25943344, 1.22610999],
                                    [6.32054686, 1.29985921],
                                    [6.37902832, 1.37559284],
                                    [6.43458939, 1.45348884],
                                    [6.486982  , 1.53368286],
                                    [6.53599479, 1.61627174],
                                    [6.58145193, 1.70131552],
                                    [6.62321754, 1.78883614],
                                    [6.66117834, 1.87883058],
                                    [6.69524962, 1.97126783],
                                    [6.72538052, 2.06608741],
                                    [6.75153636, 2.16321133],
                                    [6.77368838, 2.26254914],
                                    [6.79184036, 2.36398226],
                                    [6.80603282, 2.46736262],
                                    [6.81628855, 2.57253571],
                                    [6.82265331, 2.67931083],
                                    [6.82516461, 2.78744944],
                                    [6.82388914, 2.89623692],
                                    [6.81845397, 3.00441908],
                                    [6.80812357, 3.11148411],
                                    [6.79224225, 3.21692479],
                                    [6.77021963, 3.32016502],
                                    [6.74158488, 3.42055719],
                                    [6.70603273, 3.51740329],
                                    [6.66341655, 3.60996763],
                                    [6.61374013, 3.69749266],
                                    [6.55719139, 3.77925844],
                                    [6.49407316, 3.85456771],
                                    [6.4248121 , 3.92278659],
                                    [6.34993696, 3.98336536],
                                    [6.27005918, 4.03586273],
                                    [6.1858345 , 4.07993569],
                                    [6.09794931, 4.11536163],
                                    [6.00709607, 4.14203777],
                                    [5.9139459 , 4.15995166],
                                    [5.81913858, 4.16918755],
                                    [5.72327028, 4.16992468],
                                    [5.62688151, 4.16240281],
                                    [5.5304549 , 4.14691459],
                                    [5.43440457, 4.12389131],
                                    [5.33906833, 4.09379932],
                                    [5.24470992, 4.05714592],
                                    [5.15150773, 4.01452946],
                                    [5.05955427, 3.96660563],
                                    [4.96886832, 3.91404261],
                                    [4.87496713, 3.86429072],
                                    [4.77973767, 3.81884659],
                                    [4.68324311, 3.7780149 ],
                                    [4.5855699 , 3.74208434],
                                    [4.48683523, 3.71128343],
                                    [4.3871767 , 3.68580145],
                                    [4.28675106, 3.66576673],
                                    [4.18572543, 3.65127177],
                                    [4.08427488, 3.64236017],
                                    [3.98257685, 3.63906273],
                                    [3.88081122, 3.64138292],
                                    [3.7791582 , 3.64929429],
                                    [3.6777954 , 3.66273576],
                                    [3.5768971 , 3.68163334],
                                    [3.47663166, 3.70588047],
                                    [3.37717119, 3.73540688],
                                    [3.27868027, 3.77008679],
                                    [3.18133459, 3.80983442],
                                    [3.08531286, 3.8545361 ],
                                    [2.99081661, 3.90409743],
                                    [2.89807646, 3.95839574],
                                    [2.80453179, 4.0073956 ],
                                    [2.70976064, 4.05076292],
                                    [2.61397255, 4.08784894],
                                    [2.5174468 , 4.11807187],
                                    [2.42053204, 4.14094624],
                                    [2.32363854, 4.15605113],
                                    [2.22722436, 4.16313631],
                                    [2.13178423, 4.16199481],
                                    [2.03784212, 4.15250365],
                                    [1.94593555, 4.13464204],
                                    [1.85662546, 4.10841836],
                                    [1.77045694, 4.07396935],
                                    [1.68795282, 4.03151347],
                                    [1.60962108, 3.98131272],
                                    [1.5359274 , 3.92370734],
                                    [1.46728053, 3.85910823],
                                    [1.40405711, 3.78794567],
                                    [1.34655404, 3.71071396],
                                    [1.29500194, 3.62793378],
                                    [1.24950655, 3.54018478],
                                    [1.21010445, 3.44803753],
                                    [1.1766864 , 3.35209384],
                                    [1.14903481, 3.25293834],
                                    [1.12687713, 3.15110479],
                                    [1.10981853, 3.04710736],
                                    [1.09737619, 2.94141576],
                                    [1.08899754, 2.83444691],
                                    [1.08406029, 2.72656538],
                                    [1.08192482, 2.6180729 ],
                                    [1.08191777, 2.50920264],
                                    [1.08335576, 2.40040344],
                                    [1.0872935 , 2.29203002],
                                    [1.09473798, 2.18444729],
                                    [1.10662881, 2.07811526],
                                    [1.12376724, 1.97357268],
                                    [1.14682204, 1.87143722],
                                    [1.17631758, 1.77240071],
                                    [1.21256351, 1.67718057],
                                    [1.25569192, 1.58650867],
                                    [1.30568505, 1.5011235 ],
                                    [1.36234957, 1.42171659],
                                    [1.42536847, 1.34893517],
                                    [1.49425247, 1.28327323],
                                    [1.56847476, 1.22516939],
                                    [1.64745036, 1.17495441],
                                    [1.73055054, 1.13282108],
                                    [1.81715938, 1.0988908 ],
                                    [1.90666042, 1.07315292],
                                    [1.99847407, 1.0555239 ],
                                    [2.09206132, 1.04584401]])

    # computed orl baseline
    # calc distance between agent and closest point in orl
    car_position = np.array([x, y])
    distances = np.linalg.norm(optimized_race_line - car_position, axis=1)
    min_distance = np.min(distances)  #  min distance between agent and orl (aka: closest)
    
    # calc and set closest optimized points
    closest_opt_index = np.argmin(distances)  # nearest index in orl
    previous_opt_waypoint = closest_opt_index - 1 if closest_opt_index > 0 else len(optimized_race_line) - 1
    next_opt_waypoint = closest_opt_index + 1 if closest_opt_index < len(optimized_race_line) - 1 else 0
    closest_optimized_waypoints = [previous_opt_waypoint, next_opt_waypoint]
    
    def calc_centerline_heading_diff(waypoints, closest_waypoints, agent_heading):
        def calc_heading(point1, point2, in_degrees: bool = True):
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

        center_agent_heading_diff = agent_heading_rad - center_heading_rad  # Signed difference
        return center_agent_heading_diff

    def get_intermediate_rewards(speed, progress, steps, heading, optimized_line,
                                 optimized_waypoints):

        def calc_heading_ir(heading, optimized_race_line, closest_optimized_waypoints):
            _heading_reward = 1
            yaw_diff = calc_centerline_heading_diff(optimized_race_line, closest_optimized_waypoints, heading)
            if abs(yaw_diff) < HEADING_THRESHOLD:
                _heading_reward += 5
            elif abs(yaw_diff) < (HEADING_THRESHOLD * 1.02):  # buffer% threshold
                _heading_reward *= 0.75
            return _heading_reward

        def calc_step_ir(progress, steps):
            # reward intermediate and milestone progress
            _step_ir = 0
            if steps % STEP_INTERVAL == 0 and steps > 1:
                _step_ir += 10 * abs(progress / steps)
            if round(progress) in [10, 25, 50, 75, 100]:
                _step_ir += abs(progress * 2.0)
            return _step_ir

        speed_ir = 3 if MIN_SPEED < speed < MAX_SPEED else 0
        heading_ir = calc_heading_ir(heading, optimized_line, optimized_waypoints)
        step_ir = calc_step_ir(progress, steps)
        return speed_ir + heading_ir + step_ir

    def get_speed_angle_reward(curve, speed):
        ratio_ir = 0
        if MIN_SPEED < speed < MAX_SPEED:
            ratio_ir = 3
            curve_severity = curve * 100
            speed_threshold = MAX_SPEED * (1 - curve)

            # calculate multiplier based on curve severity and speed
            if curve_severity >= 5:
                multiplier = 1.25 if speed < speed_threshold else 0.10
                ratio_ir += multiplier * (2.5 if curve_severity < 10 else 1)
            else:
                ratio_ir += 1.10

        return ratio_ir

    def get_line_proximity_reward(min_distance):
        prox_ir = 0
        if min_distance < MAX_DISTANCE:
            scaled_proximity = 1 / (1 + math.exp(4 * (min_distance - MAX_DISTANCE / 2)))
            prox_ir = 10 * scaled_proximity

            if min_distance < (MAX_DISTANCE / 1.5):
                prox_ir *= 2.5

            # scale based on center proximity using another sigmoid
            center_proximity = 1 / (1 + math.exp(2 * (distance_from_center - (track_width * 0.3) / 2)))
            prox_ir *= 1 + 0.15 * center_proximity
        else:
            prox_ir = 1e-3
        return max(prox_ir, 1e-3)

    def calc_curvature(optimized_race_line, starting_index, num_points):
        def curvature(x1, y1, x2, y2, x3, y3, tolerance=1e-5):
            numerator = 2 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            denominator = np.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2) * ((x3 - x2) ** 2 + (y3 - y2) ** 2) * ((x3 - x1) ** 2 + (y3 - y1) ** 2))

            return numerator / max(denominator, tolerance)

        curvatures = []

        for i in range(starting_index, starting_index + num_points - 2):
            i1 = i % len(optimized_race_line)
            i2 = (i + 1) % len(optimized_race_line)
            i3 = (i + 2) % len(optimized_race_line)

            x1, y1 = optimized_race_line[i1]
            x2, y2 = optimized_race_line[i2]
            x3, y3 = optimized_race_line[i3]

            curv = curvature(x1, y1, x2, y2, x3, y3)
            curvatures.append(curv)

        return curvatures

    def calc_normalized_curve(optimized_race_line, starting_index, look_ahead=LOOK_AHEAD):
        # return average curvature between # of look_ahead points. between 0 - 1
        curvatures = calc_curvature(optimized_race_line, starting_index, num_points=look_ahead)
        average_curve = np.mean(curvatures)
        return average_curve / (average_curve + 1)

    # calc normalized curve; avg angles, returning float 0.00 to 1.00
    normalized_curve = calc_normalized_curve(optimized_race_line, closest_opt_index, LOOK_AHEAD)

    # Apply intermediate, base reward values
    reward += get_intermediate_rewards(speed, progress, steps, heading, optimized_race_line,
                                       closest_optimized_waypoints)

    # Apply reward for speed:angle ratio
    speed_angle_reward = get_speed_angle_reward(normalized_curve, speed)
    if speed_angle_reward == 0:
        reward *= 0.50
    else:
        reward += speed_angle_reward

    # Apply reward for proximity to optimized race line
    line_proximity_reward = get_line_proximity_reward(min_distance)
    if line_proximity_reward == 0:
        reward *= 0.45
    else:
        reward += line_proximity_reward

    if is_offtrack or is_crashed or not all_wheels_on_track:
        reward = 1e-3

    return max(float(reward), 1e-3)
