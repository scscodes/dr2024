def reward_function(params):
    # Extract necessary parameters
    all_wheels_on_track = params['all_wheels_on_track']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    is_reversed = params['is_reversed']
    progress = params['progress']
    speed = params['speed']
    steering_angle = abs(params['steering_angle'])

    # Initialize reward
    reward = 1.0

    # Reward if all wheels are on track
    if not all_wheels_on_track:
        return 1e-3

    # Reward for driving in reverse
    if is_reversed:
        reward += 2.0
    else:
        reward -= 1.0

    # Reward for progress
    reward += progress / 100.0

    # Reward for staying close to the track center
    reward += max(1e-3, (1 - (distance_from_center / (track_width / 2.0))) ** 2)

    # Penalize high steering angles
    if steering_angle > 15:
        reward *= 0.8

    # Reward for maintaining a reasonable speed
    if speed > 1.0:
        reward *= 1.2
    elif speed < 0.5:
        reward *= 0.8

    return float(reward)
