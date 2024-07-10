import math
def reward_function(params):
    ############ INIT ############
    reward = 1.0
    MIN_SPEED = 0.5
    MAX_SPEED = 2
    STEERING_ANGLE_THRESHOLD = 15.0  #  +/- degrees
    DIRECTION_THRESHOLD = 15.0  #  +/- degrees
    STEP_INTERVAL = 25 
    STEPS = 250  # update with each track
    
    ############ HELPER FUNCTIONS ############

    def calc_track_direction(waypoints, closest_waypoints):
        next_point = waypoints[closest_waypoints[1]]
        prev_point = waypoints[closest_waypoints[0]]
        track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
        return math.degrees(track_direction)


    def calc_direction_diff(waypoints, closest_waypoints, heading):
        track_direction = calc_track_direction(waypoints, closest_waypoints)
        direction_diff = abs(track_direction - heading)
        if direction_diff > 180:
            direction_diff = 360 - direction_diff
        return direction_diff


    def calc_upcoming_angles(waypoints, closest_waypoints, heading):
        # Look ahead to the next 5 waypoints
        future_waypoints = [waypoints[(closest_waypoints[1] + i) % len(waypoints)] for i in range(1, 6)]

        # Calculate the angles of the turns
        angles = []
        for i in range(len(future_waypoints) - 1):
            track_direction = math.atan2(future_waypoints[i + 1][1] - future_waypoints[i][1], future_waypoints[i + 1][0] - future_waypoints[i][0])
            track_direction = math.degrees(track_direction)
            direction_diff = abs(track_direction - heading)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff
            angles.append(direction_diff)

        return sum(angles) / len(angles)
    
    ############ READ ALL INPUT PARAMETERS ############
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
    ############ APPLY AND RETURN REWARD ############

    # Speed
    speed_reward = 0
    if speed >= MIN_SPEED:
        speed_reward += 5
    if speed > MAX_SPEED:
        speed_reward = -1
    reward += speed_reward
    
    # Steps (step = action taken by agent)
    step_reward = 0
    if (steps % STEP_INTERVAL) == 0:
        if progress >= (steps / STEPS) * 100:
            step_reward = 20
    reward += step_reward
    
    # Turning
    turning_reward = 0
    mean_angle = calc_upcoming_angles(waypoints, closest_waypoints, heading)
    heading_diff = abs(heading - mean_angle)
    if heading_diff <= 5 and distance_from_center >= (track_width / 2):
        turning_reward = 10
    reward += turning_reward

    # Penalize excessive steering angle
    if abs(steering_angle) > STEERING_ANGLE_THRESHOLD:
        reward *= 0.8

    # Penalize negative orientation
    current_direction_diff = calc_direction_diff(waypoints, closest_waypoints, heading)
    if current_direction_diff > DIRECTION_THRESHOLD:
        reward = 1e-3
    
    # Penalize negative termination states
    if is_crashed or is_offtrack:
        reward = 1e-3

    # Ensure the reward is positive
    reward = max(reward, 1e-3)

    return float(reward)
