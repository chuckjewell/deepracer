```python
import math

def reward_function(on_track, x, y, distance_from_center, car_orientation, progress, steps, throttle, steering, track_width, waypoints, closest_waypoint):
    '''
    @on_track (boolean) :: The vehicle is off-track if the front of the vehicle is outside of the white
    lines
    @x (float range: [0, 1]) :: Fraction of where the car is along the x-axis. 1 indicates
    max 'x' value in the coordinate system.
    @y (float range: [0, 1]) :: Fraction of where the car is along the y-axis. 1 indicates
    max 'y' value in the coordinate system.
    @distance_from_center (float [0, track_width/2]) :: Displacement from the center line of the track
    as defined by way points
    @car_orientation (float: [-3.14, 3.14]) :: yaw of the car with respect to the car's x-axis in
    radians
    @progress (float: [0,1]) :: % of track complete
    @steps (int) :: numbers of steps completed
    @throttle :: (float) 0 to 1 (0 indicates stop, 1 max throttle)
    @steering :: (float) -1 to 1 (-1 is right, 1 is left)
    @track_width (float) :: width of the track (> 0)
    @waypoints (ordered list) :: list of waypoint in order; each waypoint is a set of coordinates
    (x,y,yaw) that define a turning point
    @closest_waypoint (int) :: index of the closest waypoint (0-indexed) given the car's x,y
    position as measured by the eucliedean distance
    @@output: @reward (float [-1e5, 1e5])
    '''

    # Heavily penalize going off-track
    if not on_track:
        return float(-100.0)

    # Big bonus for completing the track quickly
    if progress >= 0.999:
        return float(10000.0 / max(steps, 1))

    # Start with a base reward
    reward = 1.0

    # Compute signed distance from center
    num_waypoints = len(waypoints)
    closest_idx = closest_waypoint
    wp = waypoints[closest_idx]
    wp_x, wp_y, wp_yaw = wp[0], wp[1], wp[2]
    dx = x - wp_x
    dy = y - wp_y
    # Perpendicular vector to the left (CCW)
    perp_x = -math.sin(wp_yaw)
    perp_y = math.cos(wp_yaw)
    signed_dfc = dx * perp_x + dy * perp_y

    # Compute signed curvature for racing line optimization
    prev_idx = (closest_idx - 1) % num_waypoints
    next_idx = (closest_idx + 1) % num_waypoints
    point_prev = waypoints[prev_idx][:2]
    point_curr = waypoints[closest_idx][:2]
    point_next = waypoints[next_idx][:2]
    vec1_x = point_curr[0] - point_prev[0]
    vec1_y = point_curr[1] - point_prev[1]
    dir1 = math.atan2(vec1_y, vec1_x)
    vec2_x = point_next[0] - point_curr[0]
    vec2_y = point_next[1] - point_curr[1]
    dir2 = math.atan2(vec2_y, vec2_x)
    delta = dir2 - dir1
    delta = ((delta + math.pi) % (2 * math.pi)) - math.pi
    len1 = math.sqrt(vec1_x**2 + vec1_y**2)
    curvature = delta / len1 if len1 > 0 else 0

    # Calculate target signed offset for optimal racing line (shift towards inside of turn)
    TURN_THRESHOLD = math.radians(5)
    MAX_OFFSET = track_width * 0.4  # Adjustable factor for offset magnitude
    if abs(delta) > TURN_THRESHOLD:
        offset_factor = (abs(delta) - TURN_THRESHOLD) / (math.pi / 2 - TURN_THRESHOLD)
        target_signed = math.copysign(MAX_OFFSET * min(offset_factor, 1.0), delta)
    else:
        target_signed = 0.0

    # Effective distance from optimal line
    effective_dfc = abs(signed_dfc - target_signed)
    max_distance = track_width / 2.0
    normalized_dfc = effective_dfc / max_distance
    reward *= (1.0 - (normalized_dfc ** 2)) ** 2  # Stronger penalty for deviation

    # Encourage alignment with track direction, looking ahead
    future_wp = (closest_idx + 3) % num_waypoints
    point_future = waypoints[future_wp][:2]
    track_dir = math.atan2(point_future[1] - point_prev[1], point_future[0] - point_prev[0])
    direction_diff = abs(track_dir - car_orientation)
    if direction_diff > math.pi:
        direction_diff = 2 * math.pi - direction_diff
    direction_diff = min(direction_diff, 2 * math.pi - direction_diff)

    if direction_diff < math.radians(15):
        reward *= 1.5
    elif direction_diff < math.radians(30):
        reward *= 1.0
    else:
        reward *= 0.3

    # Encourage high throttle based on curvature (higher in straights)
    if abs(delta) < TURN_THRESHOLD:
        reward *= (0.4 + 0.6 * throttle)  # Strong encouragement in straights
    else:
        reward *= (0.6 + 0.4 * throttle)  # Milder in turns

    # Penalize excessive steering
    abs_steering = abs(steering)
    if abs_steering > 0.8:
        reward *= 0.5
    elif abs_steering > 0.5:
        reward *= 0.8

    # Add efficiency bonus
    if steps > 0:
        efficiency = (progress * 100.0) / steps
        reward += min(efficiency, 1.0)

    # Clamp reward
    reward = max(min(reward, 1e5), -1e5)

    return float(reward)
```
