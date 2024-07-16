import math
import numpy as np


def reward_function(params):
    # initialize constants; RE-GM0-SERIES; gm03; 2022_july_open
    reward = 1.0
    CRASH_PENALTY = 0.1  # base penalty
    MIN_SPEED = 1
    MAX_SPEED = 3
    OPTIMAL_SPEED = abs((MIN_SPEED + MAX_SPEED) / 2)
    STEP_INTERVAL = 5  # steps to complete before evaluation
    DIRECTION_THRESHOLD = 20.0  # +/- degrees
    LINEAR_THRESHOLD = 0.6  # acceptable diff to satisfy linear regression
    STEERING_ANGLE_THRESHOLD = 15.0  # acceptable steering angle cap

    # input parameters
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

    ############ HELPER FUNCTIONS ############
    def calc_track_direction(point1, point2):
        track_direction = math.atan2(point2[1] - point1[1], point2[0] - point1[0])
        return math.degrees(track_direction)

    def calc_direction_diff(waypoints, closest_waypoints, heading):
        track_direction = calc_track_direction(waypoints[closest_waypoints[0]], waypoints[closest_waypoints[1]])
        direction_diff = abs(track_direction - heading)
        if direction_diff > 180:
            direction_diff = 360 - direction_diff
        return direction_diff

    def calc_linear_and_slope(points, tolerance=LINEAR_THRESHOLD):
        # plot a series of points, check they are within tolerance to be considered a "straight line"
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
        upcoming_waypoint = closest_waypoints[1]
        upcoming_waypoint_index = upcoming_waypoint

        # check immediate path; if not straight, exit early
        immediate_waypoints = waypoints[upcoming_waypoint_index:upcoming_waypoint_index + 3]
        current_index = upcoming_waypoint_index + 3  # possibly more stable than hoping 1-2 points
        linear_start, slope = calc_linear_and_slope(immediate_waypoints, LINEAR_THRESHOLD)
        if not linear_start:
            return [], None, upcoming_waypoint_index

        # relatively safe to iterate remaining waypoints
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
    on_straight = len(linear_waypoints) > 0

    # Heading alignment with straights
    heading_reward = 0
    if len(linear_waypoints) > 3 and slope is not None:
        heading_reward = 20
        slope_angle = np.degrees(np.arctan(slope))
        angle_diff = abs(heading - slope_angle)
        heading_reward = heading_reward * (1 - angle_diff / 90)  # smaller diff, higher reward
        heading_reward += max(0, heading_reward) + abs(speed)
    reward += heading_reward

    # Speed
    speed_reward = 0
    speed_diff = speed - MIN_SPEED
    if speed >= MIN_SPEED and speed <= MAX_SPEED:
        speed_reward += 2
        if on_straight and len(linear_waypoints) > 3:
            speed_reward *= max(1, speed_diff)
    reward += speed_reward

    # Steps (progress/actions taken)
    step_reward = 1
    if (steps % STEP_INTERVAL) == 0:
        step_reward = progress / steps
    reward += step_reward

    # Intermediate Progress
    if progress > 25:
        reward += 10
    if progress > 50:
        reward += 20
    if progress > 75:
        reward += 30
    if progress == 100:
        reward += 100

    # Turning
    turn_reward = 0
    if not on_straight:
        turn_reward += 1
        optimal_speed_diff = 5 - abs(speed - OPTIMAL_SPEED)
        if all_wheels_on_track:
            turn_reward += 2
        turn_reward *= max(1, optimal_speed_diff)

    # Position relative to center
    marker_1 = 0.1 * track_width
    marker_2 = 0.3 * track_width
    marker_3 = 0.8 * track_width

    if distance_from_center <= marker_1:
        reward *= 1.5
    elif distance_from_center <= marker_2:
        reward *= 1.0  # neutral reward
    elif distance_from_center <= marker_3:
        reward *= 0.5
    else:
        reward = 1e-3

    # Penalize distracted driving
    if abs(steering_angle) < STEERING_ANGLE_THRESHOLD:
        reward += 10
    else:
        reward *= 0.75

    # Penalize being off the track
    if all_wheels_on_track:
        reward += 2
    else:
        reward = 1e-3

    # Penalize negative orientation
    current_direction_diff = calc_direction_diff(waypoints, closest_waypoints, heading)
    if abs(current_direction_diff) > DIRECTION_THRESHOLD:
        reward = 1e-3

    # Penalize negative termination states
    if is_crashed or is_offtrack:
        reward = 1e-3

    # Ensure the reward is positive
    reward = max(reward, 1e-3)

    return float(reward)