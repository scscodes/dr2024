import math
def reward_function(params):

    ############ HELPER FUNCTIONS ############

    def calculate_track_direction(prev_point, next_point):
        track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
        return math.degrees(track_direction)

    def calculate_direction_diff(track_direction, heading):
        direction_diff = abs(track_direction - heading)
        if direction_diff > 180:
            direction_diff = 360 - direction_diff
        return direction_diff

    def reward_for_lap_completion(params):
        progress = params['progress']
        all_wheels_on_track = params['all_wheels_on_track']
        steps = params['steps']

        if progress == 100 and all_wheels_on_track:
            return 100  # Large reward for completing a lap without going off track
        return 0

    def reward_for_faster_lap(params, previous_lap_time):
        steps = params['steps']
        if params['progress'] == 100:
            current_lap_time = steps
            if previous_lap_time and current_lap_time < previous_lap_time:
                return 50  # Reward for completing the lap faster than the previous one
        return 0


    def reward_for_optimal_turn(params):
        waypoints = params['waypoints']
        closest_waypoints = params['closest_waypoints']
        heading = params['heading']
        distance_from_center = params['distance_from_center']
        track_width = params['track_width']

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


    ############ APPLY AND RETURN REWARD ############
    # Initialize the reward with a typical value
    reward = 1.0

    # Calculate the track direction
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    track_direction = calculate_track_direction(prev_point, next_point)

    # Calculate the direction difference
    direction_diff = calculate_direction_diff(track_direction, heading)

    # Penalize if the direction difference is too large
    DIRECTION_THRESHOLD = 12.5
    if direction_diff > DIRECTION_THRESHOLD:
        reward *= 0.5

    # Large reward for completing a lap without going off track
    reward += reward_for_lap_completion(params)

    # Reward for completing the lap faster than the previous one
    # Assuming previous_lap_time is stored and accessible
    previous_lap_time = params.get('previous_lap_time', None)
    reward += reward_for_faster_lap(params, previous_lap_time)

    # Reward for optimal turning behavior
    reward += reward_for_optimal_turn(params)

    # Penalize if the car is not on the track
    if not all_wheels_on_track:
        reward = 1e-3

    # Ensure the reward is positive
    reward = max(reward, 1e-3)

    return float(reward)