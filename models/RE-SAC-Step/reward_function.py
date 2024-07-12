import math


def reward_function(params):
    ############ INIT ############
    reward = 1.0
    MIN_SPEED = 0.5
    MAX_SPEED = 2
    STEERING_ANGLE_THRESHOLD = 15.0  # +/- degrees
    DIRECTION_THRESHOLD = 15.0  # +/- degrees
    STEP_INTERVAL = 20
    STEPS = 300  # update with each track

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

    def calc_upcoming_turn_angle(waypoints, closest_waypoints):
        next_point = waypoints[closest_waypoints[1]]
        future_point = waypoints[(closest_waypoints[1] + 1) % len(waypoints)]

        current_direction = calc_track_direction(waypoints[closest_waypoints[0]], next_point)
        future_direction = calc_track_direction(next_point, future_point)

        # Calculate the turn angle
        turn_angle = future_direction - current_direction
        if turn_angle > 180:
            turn_angle -= 360
        elif turn_angle < -180:
            turn_angle += 360
        return turn_angle


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
    speed_diff = speed - MIN_SPEED
    if speed >= MIN_SPEED:
        speed_reward += 5 * max(1, speed_diff)
    if speed > MAX_SPEED:
        speed_reward = -1
    reward += speed_reward

    # Steps (step = action taken by agent)
    step_reward = 0
    if (steps % STEP_INTERVAL) == 0:
        if progress >= (steps / STEPS) * 100:
            step_reward = 25
    reward += step_reward

    # Turning
    turning_reward = 0
    turn_angle = calc_upcoming_turn_angle(waypoints, closest_waypoints)
    direction_diff = calc_direction_diff(waypoints, closest_waypoints, heading)
    if abs(turn_angle) > 10:
        if direction_diff < 10:
            turning_reward = 15
        else:
            turning_reward = -10
    reward += turning_reward

    # Pass go, collect $200
    if progress == 100:
        reward += 25

    # Penalize being off the track
    if not all_wheels_on_track:
        reward *= 0.70

    # Penalize excessive steering angle
    if abs(steering_angle) > STEERING_ANGLE_THRESHOLD:
        reward *= 0.70

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
