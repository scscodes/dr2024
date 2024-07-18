import math
import numpy as np


def reward_function(params):
    # initialize constants; RE-GM0-SERIES; gm05; console; ace super speedway
    reward = 1.0
    MIN_SPEED = 1
    MAX_SPEED = 3
    OPTIMAL_SPEED = abs((MIN_SPEED + MAX_SPEED) / 2)
    STEP_INTERVAL = 5  # steps to complete before evaluation
    DIRECTION_THRESHOLD = 20.0  # +/- degrees
    LINEAR_THRESHOLD = 0.6  # acceptable diff to satisfy linear regression
    STEERING_ANGLE_THRESHOLD = 10.0  # acceptable steering angle cap

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
        heading_reward += 10
        slope_angle = np.degrees(np.arctan(slope))
        angle_diff = abs(heading - slope_angle)
        angle_diff = np.clip(angle_diff, 0, 90)  # angle is 0-90
        heading_reward *= heading_reward * (1 - angle_diff / 90)  # smaller diff, higher reward
    reward += max(1, heading_reward)

    # Speed
    speed_reward = 0
    if MIN_SPEED <= speed <= MAX_SPEED:
        speed_reward += 3
        speed_diff = abs(speed - MIN_SPEED)
        if len(linear_waypoints) > 3:
            speed_reward += max(1, speed_diff*len(linear_waypoints))
        if len(linear_waypoints) <= 3:
            speed_reward += max(1, abs(MAX_SPEED-speed)*len(linear_waypoints))
    reward += speed_reward

    # Steps (progress/actions taken)
    step_reward = 1
    if (steps % STEP_INTERVAL) == 0:
        step_reward = progress / steps
    reward += step_reward

    # Intermediate Progress
    if progress > 5:
        reward += 1
    if progress > 10:
        reward += 3
    if progress > 25:
        reward += 5
    if progress > 50:
        reward += 10
    if progress > 75:
        reward += 15
    if progress == 100:
        reward += 50

    # Position relative to center
    marker_1 = 0.1 * track_width
    marker_2 = 0.3 * track_width
    marker_3 = 0.8 * track_width
    offset_reward = 1
    if distance_from_center <= marker_1:
        offset_reward += 1.5
    elif distance_from_center <= marker_2:
        offset_reward += 1.0  # neutral reward
    elif distance_from_center <= marker_3:
        offset_reward -= 0.5
    reward += abs(offset_reward)

    # Penalize distracted driving
    steering_reward = 0
    if abs(steering_angle) < STEERING_ANGLE_THRESHOLD:
        steering_reward += 10
    elif abs(steering_angle) < STEERING_ANGLE_THRESHOLD * 2:
        # likely turn or correction; benefit slower speed
        steering_reward += 3 * abs(MAX_SPEED - speed)
    else:
        steering_reward = 0.1
    reward += steering_reward

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
