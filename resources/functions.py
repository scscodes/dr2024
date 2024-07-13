import math
import numpy as np

# Waypoint Alignment
# closest_waypoints[0] = [x,y] = behind ; closest_waypoint[1] = [x,y] = front
next_waypoint = waypoints[closest_waypoints[1]]
previous_waypoint = waypoints[closest_waypoints[0]]

def points_are_linear(points, tolerance=0.1):
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
            return False
    return True


def find_upcoming_straight(waypoints, closest_waypoints, tolerance=0.1):
    # iterate over waypoints for linear tolerance, returning known-good and any break-inducing waypoint index
    upcoming_waypoint = closest_waypoint[1]
    upcoming_waypoint_index = waypoints[upcoming_waypoint]
    linear_waypoints = []
    
    current_index = upcoming_waypoint_index + 3 # possibly more stable logic than hoping 1-2 points are aligned
    while current_index < len(waypoints):
        # test known-good + next waypoint
        upcoming_points = linear_waypoints + [waypoints[current_index]]
    
        if points_are_linear(upcoming_points, tolerance):
            linear_waypoints.append(waypoints[current_index])
            current_index += 1
        else:
            break # incoming turn
    return linear_waypoints, current_index
    
    

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


# Helper function to calculate the angle between two points
def calculate_angle(agent_pos, waypoint):
    y_diff = waypoint[1] - agent_pos[1]
    x_diff = waypoint[0] - agent_pos[0]
    angle = math.degrees(math.atan2(y_diff, x_diff))
    return angle


# Helper function to normalize turning direction; to reward correct heading while turning
def determine_turn_direction(agent_heading, waypoint_angle):
    angle_difference = waypoint_angle - agent_heading
    # Normalize the angle to the range [-180, 180]
    angle_difference = (angle_difference + 180) % 360 - 180
    if angle_difference > 0:
        return "Left"
    elif angle_difference < 0:
        return "Right"
    else:
        return "Straight"


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
