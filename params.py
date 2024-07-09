params = {
    "all_wheels_on_track": bool,        # flag to indicate if the agent is on the track
    "x": float,                            # agent's x-coordinate in meters
    "y": float,                            # agent's y-coordinate in meters
    "closest_objects": [int, int],         # zero-based indices of the two closest objects to the agent's current position of (x, y).
    "closest_waypoints": [int, int],       # indices of the two nearest waypoints.
    "distance_from_center": float,         # distance in meters from the track center
    "is_crashed": bool,                 # Boolean flag to indicate whether the agent has crashed.
    "is_left_of_center": bool,          # Flag to indicate if the agent is on the left side to the track center or not.
    "is_offtrack": bool,                # Boolean flag to indicate whether the agent has gone off track.
    "is_reversed": bool,                # flag to indicate if the agent is driving clockwise (True) or counter clockwise (False).
    "heading": float,                      # agent's yaw in degrees
    "objects_distance": [float, ],         # list of the objects' distances in meters between 0 and track_length in relation to the starting line.
    "objects_heading": [float, ],          # list of the objects' headings in degrees between -180 and 180.
    "objects_left_of_center": [bool, ],  # list of Boolean flags indicating whether elements' objects are left of the center (True) or not (False).
    "objects_location": [(float, float),],  # list of object locations [(x,y), ...].
    "objects_speed": [float, ],            # list of the objects' speeds in meters per second.
    "progress": float,                     # percentage of track completed
    "speed": float,                        # agent's speed in meters per second (m/s)
    "steering_angle": float,               # agent's steering angle in degrees
    "steps": int,                          # number steps completed
    "track_length": float,                 # track length in meters.
    "track_width": float,                  # width of the track
    "waypoints": [(float, float), ]        # list of (x,y) as milestones along the track center
}

# all params
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