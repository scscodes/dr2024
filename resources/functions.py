import math
import numpy as np

from resources.params import closest_waypoints, waypoints

# Waypoint reminders
next_wp_index = closest_waypoints[1]  # ahead
next_waypoint = waypoints[closest_waypoints[1]]
prev_wp_index = closest_waypoints[0]  # behind
previous_waypoint = waypoints[closest_waypoints[0]]


# angle between two points, as degrees or radians
def calc_angle(point1: tuple, point2: tuple, in_degrees: bool = True) -> float:
    # return angle (heading) between two points, in degrees (easier conditions) or radians (easier math)
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    heading_rad = math.atan2(delta_y, delta_x)
    return math.degrees(heading_rad) if in_degrees else heading_rad


# linear regression; check if points are linear, find slope
LINEAR_THRESHOLD = 0.6
def calc_linear_and_slope(points: list, tolerance: float=LINEAR_THRESHOLD) -> tuple:
    """
    Cast a line between points. Check if all points are within tolerance to be
    considered linear (straight line). If so, return bool + slope of the line.
    :param points: list[(x,y)...]
    :param tolerance: float
    :return: tuple(bool, int)
    """
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

