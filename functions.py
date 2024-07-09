import math


# Calculate the track direction
def calculate_track_direction(waypoints, closest_waypoints):
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    return math.degrees(track_direction)


# Calculate the direction difference
def calculate_direction_diff(track_direction, heading):
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    return direction_diff


# Reward for completing a lap without going off track
def reward_for_lap_completion(progress, all_wheels_on_track):
    if progress == 100 and all_wheels_on_track:
        return 100  # Large reward for completing a lap without going off track
    return 0


# Reward for faster lap completion
def reward_for_faster_lap(progress, steps, previous_lap_time):
    if progress == 100:
        current_lap_time = steps
        if previous_lap_time and current_lap_time < previous_lap_time:
            return 50  # Reward for completing the lap faster than the previous one
    return 0


# Reward turn detection and desired pathing
def reward_for_correct_turn(waypoints, closest_waypoints, heading):
    # Look ahead to the next two waypoints to identify a turn
    next_point = waypoints[closest_waypoints[1]]
    future_point = waypoints[(closest_waypoints[1] + 1) % len(waypoints)]

    # Calculate directions
    current_direction = calculate_track_direction(waypoints[closest_waypoints[0]], next_point)
    future_direction = calculate_track_direction(next_point, future_point)

    # Calculate the turn angle
    turn_angle = future_direction - current_direction
    if turn_angle > 180:
        turn_angle -= 360
    elif turn_angle < -180:
        turn_angle += 360

    # Calculate the difference between the car's heading and the future direction
    direction_diff = calculate_direction_diff(future_direction, heading)

    # Reward if the car is turning in the correct direction
    maximum_turning_threshold = 15.0
    reward = 0
    if abs(turn_angle) > maximum_turning_threshold:
        if direction_diff < maximum_turning_threshold:
            reward = 10  # Reward for turning correctly
        else:
            reward = -10  # Penalize for turning incorrectly

    return reward


# Reward minimizing turn angle, penalize wider approaches
def reward_for_optimal_turn(waypoints, closest_waypoints, heading, distance_from_center, track_width):
    # Look ahead to the next 10 waypoints
    future_waypoints = [waypoints[(closest_waypoints[1] + i) % len(waypoints)] for i in range(1, 11)]

    # Calculate the angles of the turns
    angles = []
    for i in range(len(future_waypoints) - 1):
        track_direction = calculate_track_direction(future_waypoints[i], future_waypoints[i + 1])
        direction_diff = calculate_direction_diff(track_direction, heading)
        angles.append(direction_diff)

    # Reward for the smallest angle turn and penalize for swinging too far wide
    min_angle = min(angles)
    reward = max(0, 1 - (min_angle / 180))  # Reward is higher for smaller angles

    # Penalize if the car swings too far wide
    if distance_from_center >= (track_width / 2):
        reward *= 0.5  # Penalize by reducing reward if the car is too far from the center

    return reward * 10  # Scale the reward
