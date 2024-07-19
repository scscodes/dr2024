def reward_function(params):
    # Extract necessary parameters
    all_wheels_on_track = params['all_wheels_on_track']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    is_reversed = params['is_reversed']
    progress = params['progress']
    speed = params['speed']
    steering_angle = abs(params['steering_angle'])
    is_offtrack = params['is_offtrack']
    is_crashed = params['is_crashed']

    # Initialize reward
    reward = 1.0

    # Reward if all wheels are on track
    if not all_wheels_on_track:
        reward *= 0.10

    if is_crashed or is_offtrack:
        reward = 1e-3
        return reward

    # Be meta
    if is_reversed:
        reward += progress / 100.0
        reward *= 1.5
    else:
        reward *= 0.10


    # Reward for staying close to the track center
    # reward += max(1e-3, (1 - (distance_from_center / (track_width / 2.0))) ** 2)

    # Penalize high steering angles
    # if steering_angle > 15:
    #     reward *= 0.8

    # Reward for maintaining a reasonable speed
    # if speed > 1.0:
    #     reward *= 1.2
    # elif speed < 0.5:
    #     reward *= 0.8

    
    return float(reward)
