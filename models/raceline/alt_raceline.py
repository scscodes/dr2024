import math
import numpy as np

def reward_function(params):
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
    ################## INPUT PARAMETERS ###################
    MAX_SPEED = 4
    MIN_SPEED = 1.5
    HALF_SPEED = (MAX_SPEED + MIN_SPEED)/2
    LOOKAHEAD = 8
    LINEAR_TOLERANCE = 0.1  # acceptable variance to be considered linear

    #################### RACING LINE ######################
    orl = [[ 5.04771855e+00,  6.58627988e-01],
           [ 5.04770537e+00,  8.41771377e-01],
           [ 5.04769313e+00,  1.00937050e+00],
           [ 5.04705158e+00,  1.19409169e+00],
           [ 5.04291648e+00,  1.47189799e+00],
           [ 5.03190660e+00,  1.76492790e+00],
           [ 5.01149011e+00,  2.05700512e+00],
           [ 4.97950988e+00,  2.34448219e+00],
           [ 4.93408910e+00,  2.62564070e+00],
           [ 4.87363040e+00,  2.89904578e+00],
           [ 4.79678169e+00,  3.16329930e+00],
           [ 4.70237665e+00,  3.41696374e+00],
           [ 4.58936140e+00,  3.65848172e+00],
           [ 4.45680512e+00,  3.88612975e+00],
           [ 4.30384311e+00,  4.09789394e+00],
           [ 4.12913111e+00,  4.29077826e+00],
           [ 3.93530459e+00,  4.46508094e+00],
           [ 3.72455099e+00,  4.62117163e+00],
           [ 3.49872714e+00,  4.75944152e+00],
           [ 3.25947034e+00,  4.88031742e+00],
           [ 3.00827085e+00,  4.98428694e+00],
           [ 2.74654089e+00,  5.07196096e+00],
           [ 2.47564108e+00,  5.14410681e+00],
           [ 2.19689874e+00,  5.20168849e+00],
           [ 1.91160407e+00,  5.24588229e+00],
           [ 1.62100795e+00,  5.27810539e+00],
           [ 1.32629492e+00,  5.29999949e+00],
           [ 1.02855460e+00,  5.31340259e+00],
           [ 7.28741600e-01,  5.32026055e+00],
           [ 4.27679643e-01,  5.32264495e+00],
           [ 1.26034662e-01,  5.32268765e+00],
           [-1.74982652e-01,  5.31829275e+00],
           [-4.74801657e-01,  5.30686508e+00],
           [-7.72642809e-01,  5.28589075e+00],
           [-1.06746198e+00,  5.25299535e+00],
           [-1.35790562e+00,  5.20600945e+00],
           [-1.64229029e+00,  5.14304573e+00],
           [-1.91861166e+00,  5.06257550e+00],
           [-2.18456335e+00,  4.96346649e+00],
           [-2.43761959e+00,  4.84508892e+00],
           [-2.67512454e+00,  4.70736300e+00],
           [-2.89435161e+00,  4.55073417e+00],
           [-3.09257461e+00,  4.37615458e+00],
           [-3.26718916e+00,  4.18508934e+00],
           [-3.41562575e+00,  3.97939051e+00],
           [-3.53551944e+00,  3.76136196e+00],
           [-3.62467387e+00,  3.53369262e+00],
           [-3.68099449e+00,  3.29944275e+00],
           [-3.70242820e+00,  3.06209851e+00],
           [-3.68693671e+00,  2.82569660e+00],
           [-3.63243531e+00,  2.59505304e+00],
           [-3.53694905e+00,  2.37614100e+00],
           [-3.39891914e+00,  2.17682998e+00],
           [-3.22995997e+00,  1.99727649e+00],
           [-3.03434684e+00,  1.83871185e+00],
           [-2.81588383e+00,  1.70155763e+00],
           [-2.57813133e+00,  1.58541049e+00],
           [-2.32445890e+00,  1.48902951e+00],
           [-2.05807505e+00,  1.41031619e+00],
           [-1.78206657e+00,  1.34629467e+00],
           [-1.49937393e+00,  1.29325040e+00],
           [-1.21285943e+00,  1.24681146e+00],
           [-9.32059405e-01,  1.19646371e+00],
           [-6.57600867e-01,  1.13672692e+00],
           [-3.93062579e-01,  1.06330264e+00],
           [-1.41980806e-01,  9.72912582e-01],
           [ 9.21720327e-02,  8.63383761e-01],
           [ 3.05969989e-01,  7.33617058e-01],
           [ 4.95964983e-01,  5.83489434e-01],
           [ 6.58637262e-01,  4.13798735e-01],
           [ 7.90310802e-01,  2.26231204e-01],
           [ 8.86956640e-01,  2.33878288e-02],
           [ 9.44247200e-01, -1.90921903e-01],
           [ 9.57486092e-01, -4.11259955e-01],
           [ 9.21983517e-01, -6.29622540e-01],
           [ 8.47545129e-01, -8.39917057e-01],
           [ 7.37702059e-01, -1.03812236e+00],
           [ 5.95594951e-01, -1.22110595e+00],
           [ 4.24143419e-01, -1.38635151e+00],
           [ 2.26213492e-01, -1.53189007e+00],
           [ 4.68818954e-03, -1.65629715e+00],
           [-2.37439565e-01, -1.75879759e+00],
           [-4.97022951e-01, -1.83942927e+00],
           [-7.70794103e-01, -1.89920070e+00],
           [-1.05548218e+00, -1.94017240e+00],
           [-1.34805784e+00, -1.96524716e+00],
           [-1.64576815e+00, -1.97822350e+00],
           [-1.94628298e+00, -1.98353201e+00],
           [-2.24259504e+00, -1.99903160e+00],
           [-2.53217144e+00, -2.02825669e+00],
           [-2.81235958e+00, -2.07401316e+00],
           [-3.08045360e+00, -2.13830677e+00],
           [-3.33379432e+00, -2.22232607e+00],
           [-3.56985974e+00, -2.32647676e+00],
           [-3.78623767e+00, -2.45054462e+00],
           [-3.98061606e+00, -2.59378319e+00],
           [-4.15067626e+00, -2.75503779e+00],
           [-4.29409099e+00, -2.93273230e+00],
           [-4.40835894e+00, -3.12491256e+00],
           [-4.49073485e+00, -3.32913719e+00],
           [-4.53800495e+00, -3.54230535e+00],
           [-4.54657426e+00, -3.76020142e+00],
           [-4.51515756e+00, -3.97730394e+00],
           [-4.44776323e+00, -4.18888557e+00],
           [-4.34324002e+00, -4.38988828e+00],
           [-4.20057490e+00, -4.57390484e+00],
           [-4.02797898e+00, -4.73963412e+00],
           [-3.82867113e+00, -4.88500722e+00],
           [-3.60580994e+00, -5.00862910e+00],
           [-3.36256097e+00, -5.10981379e+00],
           [-3.10211135e+00, -5.18865633e+00],
           [-2.82768732e+00, -5.24619113e+00],
           [-2.54247848e+00, -5.28448255e+00],
           [-2.24951453e+00, -5.30661822e+00],
           [-1.95150152e+00, -5.31650516e+00],
           [-1.65075319e+00, -5.31872020e+00],
           [-1.34910933e+00, -5.31817436e+00],
           [-1.04746495e+00, -5.31761048e+00],
           [-7.45820463e-01, -5.31703806e+00],
           [-4.44175973e-01, -5.31646479e+00],
           [-1.42531499e-01, -5.31589198e+00],
           [ 1.59112976e-01, -5.31531935e+00],
           [ 4.60757458e-01, -5.31474725e+00],
           [ 7.62401946e-01, -5.31417604e+00],
           [ 1.06373604e+00, -5.31279060e+00],
           [ 1.36361788e+00, -5.30766414e+00],
           [ 1.66079425e+00, -5.29599565e+00],
           [ 1.95391579e+00, -5.27519682e+00],
           [ 2.24158101e+00, -5.24296665e+00],
           [ 2.52236746e+00, -5.19732502e+00],
           [ 2.79483817e+00, -5.13659734e+00],
           [ 3.05755539e+00, -5.05941636e+00],
           [ 3.30906025e+00, -4.96467090e+00],
           [ 3.54784836e+00, -4.85146549e+00],
           [ 3.77217091e+00, -4.71887848e+00],
           [ 3.98005318e+00, -4.56608889e+00],
           [ 4.16912168e+00, -4.39229723e+00],
           [ 4.33632484e+00, -4.19670329e+00],
           [ 4.48352051e+00, -3.98326675e+00],
           [ 4.61127234e+00, -3.75405895e+00],
           [ 4.72018238e+00, -3.51086779e+00],
           [ 4.81099328e+00, -3.25534693e+00],
           [ 4.88457550e+00, -2.98903729e+00],
           [ 4.94204509e+00, -2.71345283e+00],
           [ 4.98478903e+00, -2.43008386e+00],
           [ 5.01450200e+00, -2.14039976e+00],
           [ 5.03322233e+00, -1.84584297e+00],
           [ 5.04327302e+00, -1.54777937e+00],
           [ 5.04722954e+00, -1.24746877e+00],
           [ 5.04781384e+00, -9.46016240e-01],
           [ 5.04780644e+00, -6.44371530e-01],
           [ 5.04779003e+00, -3.42726629e-01],
           [ 5.04776880e+00, -4.10816295e-02],
           [ 5.04775024e+00,  2.15035250e-01],
           [ 5.04773324e+00,  4.52712835e-01],
           [ 5.04771855e+00,  6.58627988e-01]]

    #################### FUNCTIONS ######################
    def dist_2_points(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def get_nearest_orl_point(cp, waypoints, orl):
        next_wp = waypoints[cp[1]]  # ahead
        prev_wp = waypoints[cp[0]]  # behind

        nearest_orl_point = None
        min_distance = float('inf')

        for orl_point in orl:
            dist_to_behind = dist_2_points(orl_point, prev_wp)
            dist_to_ahead = dist_2_points(orl_point, next_wp)
            min_dist_to_waypoint = min(dist_to_behind, dist_to_ahead)

            if min_dist_to_waypoint < min_distance:
                min_distance = min_dist_to_waypoint
                nearest_orl_point = orl_point
        return nearest_orl_point

    def get_distance_to_nearest_orl_point(agent_coords, cp, waypoints, orl):
        nearest_orl_point = get_nearest_orl_point(cp, waypoints, orl)
        return dist_2_points(agent_coords, nearest_orl_point)

    def fit_circle(points):
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        A = np.c_[x, y, np.ones(len(x))]
        b = x**2 + y**2

        C = np.linalg.lstsq(A, b, rcond=None)[0]
        cx = C[0] / 2
        cy = C[1] / 2
        radius = math.sqrt((C[2] + cx**2 + cy**2))

        return (cx, cy), radius

    def is_turn_left(center, agent_position):
        cx, cy = center
        ax, ay = agent_position
        vector_to_center = np.array([cx - ax, cy - ay])
        return vector_to_center[0] < 0  # True for left, False for right

    def is_linear(points, tolerance=LINEAR_TOLERANCE):
        x_coordinates = [point[0] for point in points]
        y_coordinates = [point[1] for point in points]

        A = np.vstack([x_coordinates, np.ones(len(x_coordinates))]).T
        m, c = np.linalg.lstsq(A, y_coordinates, rcond=None)[0]

        for x, y in zip(x_coordinates, y_coordinates):
            y_int = m * x + c
            distance = abs(y - y_int)
            if distance > tolerance:
                return False
        return True

    def section_is_linear(orl_index, lookahead, orl, tolerance=LINEAR_TOLERANCE):
        upcoming_orl_points = [orl[(orl_index + i) % len(orl)] for i in range(lookahead)]
        return is_linear(upcoming_orl_points, tolerance)

    #################### SETUP ######################
    next_orl_point = get_nearest_orl_point(closest_waypoints, waypoints, orl)
    next_orl_point_index = orl.index(next_orl_point)
    dist_to_orl_point = get_distance_to_nearest_orl_point((x, y), closest_waypoints, waypoints, orl)

    lookahead_points = [orl[(next_orl_point_index + i) % len(orl)] for i in range(LOOKAHEAD)]
    is_linear_section = section_is_linear(next_orl_point_index, LOOKAHEAD, orl)

    center, radius = fit_circle(lookahead_points)
    turn_is_left = is_turn_left(center, (x, y))

    speed_max_diff = abs(MAX_SPEED - speed)
    speed_min_diff = abs(speed - MIN_SPEED)
    speed_half_diff = abs(speed - HALF_SPEED)

    #################### REWARD FUNCTIONS ######################
    def _r_dist_to_orl(dv=dist_to_orl_point):
        base_reward = 10.00
        if dv <= 0.10:
            return base_reward * 1.75
        elif dv <= 0.50:
            return base_reward * 1.25
        elif dv <= 2.50:
            return base_reward * 0.85
        elif dv <= 4.00:
            return base_reward * 0.50
        return 1.00

    def _r_track_utilization_tolerance(dfl=distance_from_center, tw=track_width):
        if dfl <= 0.475 * tw:
            return 1.015
        elif dfl <= 0.480 * tw:
            return 0.30
        elif dfl <= 0.485 * tw:
            return 0.20
        elif dfl <= 0.495 * tw:
            return 0.10
        return 1e-3

    def _r_is_destroying_the_car(all_on=all_wheels_on_track, wrecked=is_crashed, gone_off=is_offtrack):
        return wrecked or gone_off or not all_on

    def _r_intermediate_progress(p, s):
        if p in [25, 50, 75, 100]:
            return 2 * p
        if p > 0 and s > 0:
            return abs(2 * p / s)
        return 1e-3

    def _r_lane_choice(points_turn_left=turn_is_left, agent_is_loc=is_left_of_center):
        return points_turn_left == agent_is_loc

    def _r_speed_by_section():
        base_speed_reward = 2
        if is_linear_section:
            base_speed_reward *= speed_min_diff
        else:
            base_speed_reward *= abs(1 - speed_half_diff)
            base_speed_reward *= 2.0 if _r_lane_choice() else 1.0
        return base_speed_reward

    def _r_steering_stability(steering_angle, steering_threshold=15):
        return 2.5 if abs(steering_angle) < steering_threshold else 1.0

    ################ REWARD AND PUNISHMENT ################
    reward = 1
    reward += _r_dist_to_orl()
    reward *= _r_track_utilization_tolerance()
    reward += _r_intermediate_progress(p=progress, s=steps)
    reward += _r_speed_by_section()
    reward += _r_steering_stability(steering_angle=steering_angle)

    if speed < MIN_SPEED:
        reward *= 0.50

    if _r_is_destroying_the_car():
        reward = 1e-3

    return float(reward)

