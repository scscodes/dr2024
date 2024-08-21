import math
import numpy as np


def reward_function(params):
    # initialize constants
    reward = 1.0
    MIN_SPEED = 0.5
    MAX_SPEED = 2
    STEP_INTERVAL = 5
    DIRECTION_THRESHOLD = 25.0
    
    # input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
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
    def calculate_turn_classification(waypoints, closest_waypoints):
        point1 = waypoints[closest_waypoints[0] - 2]
        point2 = waypoints[closest_waypoints[0] - 1]
        point3 = waypoints[closest_waypoints[0]]
        point4 = waypoints[closest_waypoints[1]]

        angle1 = calc_turn_angle(point1, point2, point3)
        angle2 = calc_turn_angle(point2, point3, point4)

        avg_angle = (angle1 + angle2) / 2

        if avg_angle < 2:
            return "Straight"
        elif avg_angle < 10:
            return "Broad Turn"
        elif avg_angle < 20:
            return "Moderate Turn"
        else:
            return "Severe Turn"

    def calc_turn_angle(point1, point2, point3):
        vector1 = np.array(point2) - np.array(point1)
        vector2 = np.array(point3) - np.array(point2)
        
        unit_vector1 = vector1 / np.linalg.norm(vector1)
        unit_vector2 = vector2 / np.linalg.norm(vector2)
        
        dot_product = np.dot(unit_vector1, unit_vector2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle = np.arccos(dot_product)
        return np.degrees(angle)
    
    # Classify the current track section
    track_classification = calculate_turn_classification(waypoints, closest_waypoints)
    
    # Apply rewards based on the classification
    if track_classification == "Straight":
        # Reward higher speeds on straight sections, scaling with difference from MAX_SPEED
        speed_diff = MAX_SPEED - speed
        reward += max(1, 5 - speed_diff)

    elif track_classification == "Broad Turn":
        # Encourage moderate speed and smooth steering
        if speed > MAX_SPEED * 0.6 or abs(steering_angle) > 10:
            reward *= 0.2  # Penalize excessive speed and steering in broad turns
        else:
            reward += 6  # Reward smooth handling

    elif track_classification == "Moderate Turn":
        # Encourage slower speeds and careful navigation
        if speed > MAX_SPEED * 0.4 or abs(steering_angle) > 15:
            reward *= .15  # Penalize aggressive driving
        else:
            reward += 8  # Reward careful handling

    elif track_classification == "Severe Turn":
        # Encourage very slow speeds and precise navigation
        if speed > MAX_SPEED * 0.2 or abs(steering_angle) > 20:
            reward *= 0.01  # Heavily penalize excessive speed and steering in severe turns
        else:
            reward += 10  # Reward safe handling
    
    # Reward itermediate progress
    if (steps % STEP_INTERVAL) == 0:
        step_reward = progress / steps
        reward += 1.5 * step_reward

    # Intermediate Progress
    progress_reward = abs(progress/2)
    reward += progress_reward
    
    # Penalize going off-track or crashing
    if is_offtrack or is_crashed:
        reward = 1e-3  # Minimal reward for negative outcomes

    # Reward staying on the track
    if all_wheels_on_track:
        reward *= 1.10
    else:
        reward *= 0.25
    
    # Ensure reward remains positive
    reward = max(reward, 1e-3)

    return float(reward)
