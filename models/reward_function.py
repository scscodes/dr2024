import math
import numpy as np


def reward_function(params):
    # initialize constants; REGMSERIES;
    reward = 1.0
    MIN_SPEED = 0.5
    MAX_SPEED = 3
    OPTIMAL_SPEED = abs((MIN_SPEED + MAX_SPEED) / 2)
    STEP_INTERVAL = 5  # steps to complete before evaluation
    DIRECTION_THRESHOLD = 25.0  # +/- degrees

    LINEAR_THRESHOLD = 0.6  # acceptable diff to satisfy linear regression
    STEERING_ANGLE_THRESHOLD = 15.0  # acceptable steering angle cap

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

    ############ HELPER FUNCTIONS ############
    def calc_heading(point1, point2, in_degrees: bool = True):
        # return a heading in degrees (easier conditional) or radians (easier math)
        delta_y = point2[1] - point1[1]
        delta_x = point2[0] - point1[0]
        heading_rad = math.atan2(delta_y, delta_x)
        return math.degrees(heading_rad) if in_degrees else heading_rad

    def calc_centerline_heading_diff(waypoints, closest_waypoints, agent_heading):
        # return difference in heading value, between agent and centerline
        agent_heading_rad = math.radians(agent_heading)
        if closest_waypoints[1] < len(waypoints) - 1:
            center_heading_rad = calc_heading(waypoints[closest_waypoints[0]], waypoints[closest_waypoints[1]], False)
        else:
            center_heading_rad = calc_heading(waypoints[closest_waypoints[0]], waypoints[closest_waypoints[0] - 1], False)

        center_agent_heading_diff = abs(agent_heading_rad - center_heading_rad)
        return center_agent_heading_diff

    def calc_turn_angle(waypoints, current_index, linear_waypoints):
        current_slope = calc_linear_and_slope(linear_waypoints)[1]
        next_index = (current_index + 3) % len(waypoints)
        if next_index < len(waypoints):
            next_slope = calc_linear_and_slope(waypoints[current_index:next_index + 1])[1]
            angle = abs(math.degrees(math.atan(next_slope - current_slope)))
            return angle
        return 0

    def calc_linear_and_slope(points, tolerance=LINEAR_THRESHOLD):
        # check that sequential points are within tolerance, considered a "straight line"
        x_coordinates = [point[0] for point in points]
        y_coordinates = [point[1] for point in points]
        # linear regression; flatten coordinates and find appropriate line
        A = np.vstack([x_coordinates, np.ones(len(x_coordinates))]).T
        m, c = np.linalg.lstsq(A, y_coordinates, rcond=None)[0]

        # calc distance of each point to the line
        for x, y in zip(x_coordinates, y_coordinates):
            y_int = m * x + c
            distance = abs(y - y_int)
            if distance > tolerance:
                return False, None
        return True, m

    def parse_upcoming_waypoints(waypoints, closest_waypoints, tolerance=LINEAR_THRESHOLD):
        # iterate over waypoints for linear tolerance, return linear list, slope, and breakpoint index
        next_index = closest_waypoints[1]
        immediate_waypoints = waypoints[next_index:next_index + 3]

        linear_start, slope = calc_linear_and_slope(immediate_waypoints, LINEAR_THRESHOLD)
        if not linear_start:
            return [], None, next_index

        # relatively safe to iterate remaining waypoints
        current_index = next_index + 3  # more stable than 1-2 points
        linear_waypoints = list(immediate_waypoints)
        while current_index < len(waypoints):
            upcoming_points = linear_waypoints + [waypoints[current_index]]
            upcoming_points_are_linear, slope = calc_linear_and_slope(upcoming_points, tolerance)
            if upcoming_points_are_linear:
                linear_waypoints.append(waypoints[current_index])
                current_index += 1
            else:
                break  # incoming turn
        return linear_waypoints, slope, current_index

    ############ APPLY AND RETURN REWARD ############

    linear_waypoints, slope, corner_index = parse_upcoming_waypoints(waypoints, closest_waypoints)
    on_straight = len(linear_waypoints) > 5

    # Speed
    speed_reward = 0
    if MIN_SPEED <= speed <= MAX_SPEED:
        speed_reward += 3
        speed_diff = abs(speed - MIN_SPEED)
        if on_straight:
            speed_reward += max(1, speed_diff*len(linear_waypoints))
        if not on_straight:
            speed_reward += max(1, abs(MAX_SPEED-speed)*len(linear_waypoints))
    else:
        speed_reward -= 5
    reward += speed_reward

    # Turning; reward navigation, but penalize excessive speeds
    turn_angle_reward = 0
    if not on_straight:
        turn_angle_reward += 1
        turn_angle = calc_turn_angle(waypoints, corner_index, linear_waypoints)
        if turn_angle > 90:
            turn_angle_reward += 10
            if speed > (MAX_SPEED * 0.40):
                turn_angle_reward *= 0.40
        elif turn_angle > 45:
            turn_angle_reward += 5
            if speed > (MAX_SPEED * 0.60):
                turn_angle_reward *= 0.60
        elif turn_angle > 10:
            turn_angle_reward += 2
            if speed > (MAX_SPEED * 0.90):
                turn_angle_reward *= 0.90
    reward += turn_angle_reward

    # Steps (progress/actions taken)
    step_reward = 1
    if (steps % STEP_INTERVAL) == 0:
        step_reward = progress / steps
    reward += step_reward

    # Intermediate Progress
    progress_reward = abs(progress/2)
    reward += progress_reward

    # Position relative to center
    offset_reward = 0
    if distance_from_center <= track_width * 0.05 and on_straight:
        heading_multiplier = abs(4 * calc_centerline_heading_diff(waypoints, closest_waypoints, heading))
        speed_multiplier = abs(2 * (MAX_SPEED/speed))
        offset_reward += 2 + heading_multiplier + speed_multiplier
    if distance_from_center <= track_width * 0.10:
        offset_reward += 1.5
    elif distance_from_center <= track_width * 0.20:
        offset_reward += 1.0  # neutral reward
    elif distance_from_center <= track_width * 0.25:
        offset_reward += 0.5
    else:
        offset_reward -= 1
    reward += offset_reward

    # Penalize distracted driving
    steering_reward = 0
    if abs(steering_angle) <= STEERING_ANGLE_THRESHOLD:
        steering_reward += 5
    elif abs(steering_angle) <= STEERING_ANGLE_THRESHOLD * 1.5:
        steering_reward -= 1.5 * abs(MAX_SPEED - speed)
    elif abs(steering_angle) <= STEERING_ANGLE_THRESHOLD * 2:
        steering_reward -= 2 * abs(MAX_SPEED - speed)
    else:
        steering_reward = 0
    reward += steering_reward

    # Penalize negative orientation
    current_direction_diff = calc_centerline_heading_diff(waypoints, closest_waypoints, heading)
    if abs(current_direction_diff) > DIRECTION_THRESHOLD:
        reward = 1e-3
        return reward

    # Penalize negative termination states
    if is_crashed or is_offtrack:
        reward = 1e-3

    # Penalize being off the track
    if all_wheels_on_track:
        reward += 2
    else:
        reward *= 0.25

    # Ensure the reward is positive
    reward = max(reward, 1e-3)

    return float(reward)
