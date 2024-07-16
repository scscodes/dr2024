import math
import numpy as np


def reward_function(params):
    # initialize constants; RE-SAC-GM0; reverse01 default

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
    is_reversed = params['is_reversed']
    is_offtrack = params['is_offtrack']
    is_crashed = params['is_crashed']

    # Speed
    speed_reward = 0
    speed_diff = speed - MIN_SPEED
    if speed >= MIN_SPEED and speed <= MAX_SPEED:
        speed_reward += 2
        # if on_straight and len(linear_waypoints) > 3:
        #     speed_reward *= max(1, speed_diff)
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

    # Penalize negative termination states
    if is_crashed or is_offtrack:
        reward = 1e-3


    if not is_reversed:
        reward *= 0.90
    else:
        reward *= 1.20

    # Ensure the reward is positive
    reward = max(reward, 1e-3)

    return float(reward)
