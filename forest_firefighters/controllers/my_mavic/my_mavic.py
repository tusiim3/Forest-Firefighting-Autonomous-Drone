"""
my_mavic.py
Autonomous forest firefighting controller for the Mavic 2 Pro drone.
Implements: state machine, grid patrol, HSV fire detection,
GPS navigation, visual servoing, multi-fire queue, water drop.
"""

import sys
sys.path.insert(0, '/usr/local/webots/lib/controller/python39')
sys.path.insert(0, '/home/tusiim3/.local/lib/python3.9/site-packages')

import math
import numpy as np
from enum import Enum
from controller import Robot
from fire_detector import FireDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


# ---------------------------------------------------------------------------
# PID Controller
# ---------------------------------------------------------------------------

class PID:
    def __init__(self, kp, ki, kd, max_output=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / max(dt, 1e-6)
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.max_output is not None:
            output = clamp(output, -self.max_output, self.max_output)
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

class State(Enum):
    TAKEOFF       = 0
    PATROL        = 1
    APPROACH_FIRE = 2
    SERVO_CENTER  = 3
    EXTINGUISH    = 4
    DONE          = 5


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P      = 3.0
K_ROLL_P          = 50.0
K_PITCH_P         = 30.0
MAX_YAW           = 0.4
MAX_PITCH         = -1.0

TARGET_ALTITUDE   = 45.0
ARRIVAL_RADIUS    = 1.5
SERVO_THRESHOLD   = 20
WATER_COOLDOWN    = 15.0
FIRE_QUEUE_DEDUP  = 3.0


# ---------------------------------------------------------------------------
# Patrol waypoints
# ---------------------------------------------------------------------------

def generate_patrol_waypoints(x_min=5, x_max=23, y_min=5, y_max=23, n=3):
    xs = [x_min + (x_max - x_min) * i / (n - 1) for i in range(n)]
    ys = [y_min + (y_max - y_min) * j / (n - 1) for j in range(n)]
    waypoints = []
    for j, y in enumerate(ys):
        row = xs if j % 2 == 0 else list(reversed(xs))
        for x in row:
            waypoints.append([x, y])
    return waypoints


# ---------------------------------------------------------------------------
# GPS / navigation helpers
# ---------------------------------------------------------------------------

def distance_2d(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def estimate_fire_gps(cx_px, cy_px, cam_w, cam_h,
                      drone_x, drone_y, altitude, hfov_deg=69.0):
    hfov = math.radians(hfov_deg)
    vfov = hfov * (cam_h / cam_w)
    offset_x = (cx_px / cam_w - 0.5) * 2 * altitude * math.tan(hfov / 2)
    offset_y = (cy_px / cam_h - 0.5) * 2 * altitude * math.tan(vfov / 2)
    return drone_x + offset_x, drone_y + offset_y


# ---------------------------------------------------------------------------
# Main Robot Class
# ---------------------------------------------------------------------------

class MyMavic(Robot):

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())

        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)

        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)

        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)

        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.camera_pitch = self.getDevice("camera pitch")
        self.camera_pitch.setPosition(1.55)

        self.fl = self.getDevice("front left propeller")
        self.fr = self.getDevice("front right propeller")
        self.rl = self.getDevice("rear left propeller")
        self.rr = self.getDevice("rear right propeller")
        for m in [self.fl, self.fr, self.rl, self.rr]:
            m.setPosition(float('inf'))
            m.setVelocity(1.0)

        self.detector = FireDetector(
            self.camera.getWidth(), self.camera.getHeight())

        self.state = State.TAKEOFF

        self.waypoints = generate_patrol_waypoints()
        self.wp_idx = 0

        self.fire_queue = []
        self.fire_target = None

        self.extinguish_start = None
        self.water_active = False

        # SERVO_CENTER lost-fire tolerance counter
        self.fire_lost_count = 0

        self.x = 0.0
        self.y = 0.0
        self.altitude = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll_accel = 0.0
        self.pitch_accel = 0.0

        print("[MyMavic] Initialised. Beginning TAKEOFF sequence.")

    # -----------------------------------------------------------------------
    # Motor mixing
    # -----------------------------------------------------------------------

    def set_motors(self, roll_input, pitch_input, yaw_input, vertical_input):
        fl = K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
        fr = K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
        rl = K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
        rr = K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input
        self.fl.setVelocity(fl)
        self.fr.setVelocity(-fr)
        self.rl.setVelocity(-rl)
        self.rr.setVelocity(rr)

    # -----------------------------------------------------------------------
    # Altitude control
    # -----------------------------------------------------------------------

    def altitude_control(self):
        diff = clamp(
            TARGET_ALTITUDE - self.altitude + K_VERTICAL_OFFSET, -1, 1)
        return K_VERTICAL_P * pow(diff, 3.0)

    # -----------------------------------------------------------------------
    # Navigation
    # -----------------------------------------------------------------------

    def nav_to(self, tx, ty):
        target_angle = np.arctan2(ty - self.y, tx - self.x)
        angle_left = target_angle - self.yaw
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        
        if angle_left > np.pi:
            angle_left -= 2 * np.pi
            
        yaw_d = MAX_YAW * angle_left / (2 * np.pi)
        
        # Engage forward pitch only when roughly aligned with target trajectory
        if abs(angle_left) < 0.5:
            pitch_d = MAX_PITCH
        else:
            pitch_d = 0.0
            
        return yaw_d, pitch_d

    def arrived_at(self, tx, ty):
        return distance_2d(self.x, self.y, tx, ty) < ARRIVAL_RADIUS

    # -----------------------------------------------------------------------
    # Visual servoing
    # -----------------------------------------------------------------------

    def servo_to_fire(self, cx, cy):
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        
        yaw_error = (cx - w / 2) / (w / 2)
        pitch_error = (cy - h / 2) / (h / 2)
        
        if abs(cx - w / 2) > SERVO_THRESHOLD:
            yaw_d = -MAX_YAW * yaw_error
        else:
            yaw_d = 0.0
            
        if abs(cy - h / 2) > SERVO_THRESHOLD:
            # Negative pitch_d drives the chassis forward
            pitch_d = MAX_PITCH * pitch_error 
        else:
            pitch_d = 0.0
            
        return yaw_d, pitch_d

    # -----------------------------------------------------------------------
    # Fire queue management
    # -----------------------------------------------------------------------

    def enqueue_fire(self, fx, fy):
        for ef in self.fire_queue:
            if distance_2d(ef[0], ef[1], fx, fy) < FIRE_QUEUE_DEDUP:
                return
        if self.fire_target and distance_2d(
                self.fire_target[0], self.fire_target[1], fx, fy) < FIRE_QUEUE_DEDUP:
            return
        self.fire_queue.append([fx, fy])
        print(f"[MyMavic] Fire queued at ({fx:.1f}, {fy:.1f}). Queue depth: {len(self.fire_queue)}")

    def pop_nearest_fire(self):
        if not self.fire_queue:
            return None
        self.fire_queue.sort(
            key=lambda f: distance_2d(self.x, self.y, f[0], f[1]))
        return self.fire_queue.pop(0)

    # -----------------------------------------------------------------------
    # Water drop
    # -----------------------------------------------------------------------

    def drop_water(self):
        self.setCustomData("15")
        self.water_active = True
        self.extinguish_start = self.getTime()
        print(f"[MyMavic] Water dropped at ({self.x:.1f}, {self.y:.1f})")

    # -----------------------------------------------------------------------
    # Main run loop
    # -----------------------------------------------------------------------

    def run(self):
        yaw_d   = 0.0
        pitch_d = 0.0

        while self.step(self.time_step) != -1:
            t = self.getTime()
            dt = self.time_step / 1000.0

            # --- Read sensors ---
            self.roll, self.pitch, self.yaw = self.imu.getRollPitchYaw()
            self.x, self.y, self.altitude    = self.gps.getValues()
            self.roll_accel, self.pitch_accel, _ = self.gyro.getValues()

            # --- Reset custom data each tick unless dropping ---
            if not self.water_active:
                self.setCustomData("0")

            # --- Perception (skip during water cooldown) ---
            detected, cx, cy, radius = False, 0.0, 0.0, 0.0
            if not self.water_active and self.altitude > TARGET_ALTITUDE - 1:
                detected, cx, cy, radius = self.detector.detect(self.camera)

            # --- State machine ---

            if self.state == State.TAKEOFF:
                yaw_d, pitch_d = 0.0, 0.0
                if self.altitude >= TARGET_ALTITUDE - 1:
                    print("[MyMavic] Altitude reached. Starting PATROL.")
                    self.state = State.PATROL

            elif self.state == State.PATROL:
                tx, ty = self.waypoints[self.wp_idx]
                yaw_d, pitch_d = self.nav_to(tx, ty)
                if self.arrived_at(tx, ty):
                    self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
                    print(f"[MyMavic] Waypoint reached. Next: {self.waypoints[self.wp_idx]}")
                if detected:
                    # Calculate true global coordinates of the fire
                    fx, fy = estimate_fire_gps(
                        cx, cy, 
                        self.camera.getWidth(), self.camera.getHeight(), 
                        self.x, self.y, self.altitude
                    )
                    self.fire_target = [fx, fy]
                    print(f"[MyMavic] Fire logged at GPS ({fx:.1f}, {fy:.1f}). Initiating vector approach.")
                    self.state = State.APPROACH_FIRE

            elif self.state == State.APPROACH_FIRE:
                tx, ty = self.fire_target
                yaw_d, pitch_d = self.nav_to(tx, ty)
                if self.arrived_at(tx, ty):
                    print("[MyMavic] Arrived at fire zone. Switching to SERVO_CENTER.")
                    self.fire_lost_count = 0
                    self.state = State.SERVO_CENTER

            elif self.state == State.SERVO_CENTER:
                if not detected:
                    self.fire_lost_count += 1
                    if self.fire_lost_count > 30:
                        print("[MyMavic] Fire lost. Returning to PATROL.")
                        self.fire_lost_count = 0
                        self.state = State.PATROL
                        yaw_d, pitch_d = 0.0, 0.0
                else:
                    self.fire_lost_count = 0
                    if self.detector.is_centered(cx, cy):
                        print("[MyMavic] Fire centred. Switching to EXTINGUISH.")
                        self.state = State.EXTINGUISH
                        yaw_d, pitch_d = 0.0, 0.0
                    else:
                        yaw_d, pitch_d = self.servo_to_fire(cx, cy)

            elif self.state == State.EXTINGUISH:
                yaw_d, pitch_d = 0.0, 0.0
                if self.extinguish_start is None:
                    self.drop_water()
                if t - self.extinguish_start > WATER_COOLDOWN:
                    self.water_active = False
                    self.setCustomData("0")
                    self.extinguish_start = None
                    if self.fire_queue:
                        self.fire_target = self.pop_nearest_fire()
                        print(f"[MyMavic] Next fire in queue: {self.fire_target}")
                        self.state = State.APPROACH_FIRE
                    else:
                        print("[MyMavic] All fires handled. Returning to PATROL.")
                        self.fire_target = None
                        self.state = State.PATROL

            # --- Compute stabilisation inputs ---
            roll_input  = K_ROLL_P  * clamp(self.roll,  -1, 1) + self.roll_accel
            pitch_input = K_PITCH_P * clamp(self.pitch, -1, 1) + self.pitch_accel + pitch_d
            yaw_input   = yaw_d
            vertical_input = self.altitude_control()

            self.set_motors(roll_input, pitch_input, yaw_input, vertical_input)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

import traceback
try:
    robot = MyMavic()
    robot.run()
except Exception as e:
    print("CRASH:", e)
    traceback.print_exc()