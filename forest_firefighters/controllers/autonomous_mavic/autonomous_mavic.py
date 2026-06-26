from controller import Robot
import sys
import random
import optparse

try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")

try:
    import cv2
except ImportError:
    sys.exit("Warning: 'cv2' module not found.")


def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


class Mavic(Robot):
    # constants
    K_VERTICAL_THRUST = 68.5
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0
    K_ROLL_P = 50.0
    K_PITCH_P = 30.0

    # Used by waypoint patrol.
    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1

    # Precision between the target position and the robot position in meters.
    target_precision = 0.5

    FIRE_X_CENTER_BAND = 20          # pixels: smoke is horizontally centered
    FIRE_Y_CENTER_BAND = 25          # pixels: smoke is vertically centered
    FIRE_EDGE_MARGIN = 30            # pixels: smoke is close to image boundary

    FIRE_MAX_YAW = 0.12              # gentle yaw for normal alignment
    FIRE_SMALL_YAW = 0.06            # very gentle yaw near edges/approach

    FIRE_MAX_PITCH = 0.18            # proportional forward/backward correction used outside CENTER_Y

    # CENTER_Y tuning:
    # After yaw alignment, the drone is already facing the smoke horizontally.
    # These values make the vertical centering movement slightly faster without
    # making EDGE_RECOVERY more aggressive.
    FIRE_CENTER_Y_MAX_PITCH = 0.24
    FIRE_CENTER_Y_MIN_PITCH = 0.08

    # Experiment: when CENTER_Y is active, ignore proportional pitch and force
    # forward motion. Negative pitch disturbance moves the drone forward.
    FIRE_CENTER_Y_FORCED_PITCH = -0.50

    FIRE_SLOW_FORWARD = -0.12
    FIRE_SLOW_BACKWARD = 0.12

    FIRE_CENTER_HOLD_STEPS = 4       # hold center for a few control updates
    FIRE_LOST_RECOVERY_SECONDS = 2.5 # do not instantly return to patrol
    FIRE_LOST_YAW = 0.05
    FIRE_LOST_PITCH = 0.08

    FIRE_DETECTION_INTERVAL_PATROL = 1.0
    FIRE_DETECTION_INTERVAL_TRACKING = 0.2

    SAVE_FIRE_DETECTION_IMAGE = False

    def __init__(self):
        Robot.__init__(self)

        self.time_step = int(self.getBasicTimeStep())

        self.water_to_drop = 0

        # Get and enable devices.
        self.camera = self.getDevice("camera")
        self.camera.enable(8 * self.time_step)

        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)

        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)

        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")

        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(1.55)  # vertical/downward point of view

        motors = [
            self.front_left_motor,
            self.front_right_motor,
            self.rear_left_motor,
            self.rear_right_motor,
        ]
        for motor in motors:
            motor.setPosition(float("inf"))
            motor.setVelocity(1)

        self.current_pose = 6 * [0]  # X, Y, Z, roll, pitch, yaw
        self.target_position = [0, 0, 0]
        self.target_index = 0

        self.world_fire_quadrants = [0, 0]
        self.img_coord_fire = []
        self.WaterDropStatus = False

        # ---------------------------------------------------------------------
        # New state used by the smoke-centering navigation logic.
        # ---------------------------------------------------------------------
        self.fire_mode = "PATROL"
        self.fire_target_visible = False
        self.centered_fire_steps = 0

        self.last_fire_seen_time = -1000.0
        self.last_fire_coord = []
        self.last_fire_raw_coord = []
        self.last_fire_x_error = 0.0
        self.last_fire_y_error = 0.0

        # Because get_image_from_camera() rotates the camera image, the processed
        # image dimensions may not be the same as camera.getWidth()/getHeight().
        self.processed_image_width = self.camera.getWidth()
        self.processed_image_height = self.camera.getHeight()

    # -------------------------------------------------------------------------
    # Camera and position helpers.
    # -------------------------------------------------------------------------
    def get_image_from_camera(self):
        """
        Take an image from the camera and prepare it for OpenCV processing.

        Returns:
            np.ndarray or None:
                RGB image after rotation and flip. Returns None if the camera
                frame is not available yet.
        """
        image = self.camera.getImage()
        if image is None:
            return None

        width = self.camera.getWidth()
        height = self.camera.getHeight()

        img = np.frombuffer(image, dtype=np.uint8)
        expected_length = width * height * 4
        if len(img) != expected_length:
            return None

        img = img.reshape((height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, 1)

        self.processed_image_height, self.processed_image_width = img.shape[:2]
        return img

    def get_processed_camera_size(self):
        """
        Returns the dimensions of the image actually used by OpenCV.

        This matters because the camera image is rotated before fire coordinates
        are calculated. Smoke coordinates must be compared against the processed
        image size, not blindly against the raw Webots camera width/height.
        """
        return self.processed_image_width, self.processed_image_height

    def set_position(self, pos):
        """
        Set the current absolute position and attitude of the robot.

        Parameters:
            pos (list): [X, Y, Z, roll, pitch, yaw]
        """
        self.current_pose = pos

    def set_fire_mode(self, mode, x_img=None, y_img=None, yaw_cmd=None, pitch_cmd=None):
        """
        Keep logs minimal: print only when the fire navigation mode changes.
        """
        if mode == self.fire_mode:
            return

        self.fire_mode = mode

        if x_img is None:
            print(f"[{self.getName()}] fire mode -> {mode}")
            return

        print(
            f"[{self.getName()}] fire mode -> {mode} | "
            f"smoke=({x_img:.1f}, {y_img:.1f}) | "
            f"yaw={yaw_cmd:.3f}, pitch={pitch_cmd:.3f}"
        )

    # -------------------------------------------------------------------------
    # Original waypoint patrol logic.
    # -------------------------------------------------------------------------
    def move_to_target(self, waypoints, verbose_movement=False, verbose_target=True):
        """
        Move the robot to the given patrol coordinates.

        Returns:
            yaw_disturbance (float)
            pitch_disturbance (float)
        """
        if self.target_position[0:2] == [0, 0]:
            self.target_position[0:2] = waypoints[0]
            if verbose_target:
                print(f"[{self.getName()}] First target: {self.target_position[0:2]}")

        if all(
            [
                abs(x1 - x2) < self.target_precision
                for (x1, x2) in zip(self.target_position, self.current_pose[0:2])
            ]
        ):
            self.target_index += 1
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0

            self.target_position[0:2] = waypoints[self.target_index]
            if verbose_target:
                print(f"[{self.getName()}] Target reached! New target: {self.target_position[0:2]}")

        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1],
            self.target_position[0] - self.current_pose[0],
        )

        angle_left = self.target_position[2] - self.current_pose[5]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if angle_left > np.pi:
            angle_left -= 2 * np.pi

        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)

        # Avoid log10(0), which can happen when perfectly aligned.
        safe_angle = max(abs(angle_left), 1e-6)
        pitch_disturbance = clamp(
            np.log10(safe_angle),
            self.MAX_PITCH_DISTURBANCE,
            0.1,
        )

        if verbose_movement:
            distance_left = np.sqrt(
                ((self.target_position[0] - self.current_pose[0]) ** 2)
                + ((self.target_position[1] - self.current_pose[1]) ** 2)
            )
            print(
                f"[{self.getName()}] remaining angle: {angle_left:.4f}, "
                f"remaining distance: {distance_left:.4f}"
            )

        return yaw_disturbance, pitch_disturbance

    # -------------------------------------------------------------------------
    # New smoke-centering navigation logic.
    # -------------------------------------------------------------------------
    def yaw_towards_smoke_center(self, x_error, max_yaw):
        """
        Convert horizontal image error into a gentle yaw command.

        x_error = x_img - image_center_x

        With the processed camera image:
            x_error < 0 means smoke is left of center.
            x_error > 0 means smoke is right of center.

        The sign is intentionally negative to preserve the observed useful
        behavior of the old controller: it turned in the correct direction,
        but it was too aggressive and also moved forward/backward at the same
        time.
        """
        width, _ = self.get_processed_camera_size()
        half_width = max(width / 2.0, 1.0)
        return clamp(-x_error / half_width * max_yaw, -max_yaw, max_yaw)

    def pitch_towards_smoke_center(self, y_error, max_pitch):
        """
        Convert vertical image error into forward/backward pitch correction.

        y_error = y_img - image_center_y

            y_error < 0: smoke is near the top    -> move backward slightly
            y_error > 0: smoke is near the bottom -> move forward slightly

        Important sign convention for this Webots Mavic controller:
            negative pitch disturbance = forward movement
            positive pitch disturbance = backward movement

        The minus sign below is intentional. In the previous version, CENTER_Y
        produced positive pitch when the smoke was low in the image. Visually,
        that looked like the drone had stopped or was backing away instead of
        moving forward to center the smoke.
        """
        _, height = self.get_processed_camera_size()
        half_height = max(height / 2.0, 1.0)
        return clamp(-y_error / half_height * max_pitch, -max_pitch, max_pitch)

    def log_center_y_step(
        self,
        x_img,
        y_img,
        x_error,
        y_error,
        yaw_cmd,
        pitch_cmd,
        width,
        height,
        near_left,
        near_right,
        near_top,
        near_bottom,
    ):
        """
        Print only CENTER_Y troubleshooting information.

        This log is intentionally focused on the new experiment:
            - raw OpenCV coordinate from smoke detection
            - swapped control coordinate used by the controller
            - x/y errors after the coordinate swap
            - whether EDGE_RECOVERY would have triggered
            - final yaw/pitch command
            - drone position and attitude
        """
        raw_text = "None"
        if self.last_fire_raw_coord and len(self.last_fire_raw_coord) >= 2:
            raw_text = f"({self.last_fire_raw_coord[0]:.1f}, {self.last_fire_raw_coord[1]:.1f})"

        print(
            f"[{self.getName()}] CENTER_Y override step | "
            f"time={self.getTime():.2f}s | "
            f"raw_smoke={raw_text} | "
            f"control_smoke=({x_img:.1f}, {y_img:.1f}) | "
            f"control_size=({width:.0f}x{height:.0f}) | "
            f"err=(x={x_error:.1f}, y={y_error:.1f}) | "
            f"edge_flags=(L={near_left}, R={near_right}, T={near_top}, B={near_bottom}) | "
            f"cmd=(yaw={yaw_cmd:.3f}, pitch={pitch_cmd:.3f}) | "
            f"drone_pos=({self.current_pose[0]:.2f}, {self.current_pose[1]:.2f}, {self.current_pose[2]:.2f}) | "
            f"attitude=(roll={self.current_pose[3]:.3f}, pitch={self.current_pose[4]:.3f}, yaw={self.current_pose[5]:.3f})"
        )

    def smoke_tracking_control(self):
        """
        Visual-servo controller for the smoke/fire target.

        This version tests two specific assumptions:
            1. fire_detection() returns swapped control coordinates:
                   control_x = raw_y
                   control_y = raw_x
            2. Once control X is centered, CENTER_Y overrides top/bottom
               EDGE_RECOVERY and forces pitch = -0.5.

        Why CENTER_Y is moved before EDGE_RECOVERY:
            If X is already centered, the drone should not waste time in edge
            recovery because of top/bottom Y position. It should immediately
            move forward through CENTER_Y and watch how the swapped Y coordinate
            responds.

        Returns:
            yaw_disturbance, pitch_disturbance
        """
        if not self.img_coord_fire or len(self.img_coord_fire) < 2:
            self.centered_fire_steps = 0
            self.set_fire_mode("NO_FIRE_TARGET")
            return 0.0, 0.0

        # Raw processed image dimensions after rotate+flip.
        processed_width, processed_height = self.get_processed_camera_size()

        # Because fire_detection() now returns [raw_y, raw_x], the control frame
        # has its width/height swapped relative to the processed OpenCV image.
        width = processed_height
        height = processed_width

        x_img, y_img = float(self.img_coord_fire[0]), float(self.img_coord_fire[1])

        center_x = width / 2.0
        center_y = height / 2.0

        x_error = x_img - center_x
        y_error = y_img - center_y

        self.last_fire_coord = [x_img, y_img]
        self.last_fire_x_error = x_error
        self.last_fire_y_error = y_error

        near_left = x_img < self.FIRE_EDGE_MARGIN
        near_right = x_img > width - self.FIRE_EDGE_MARGIN
        near_top = y_img < self.FIRE_EDGE_MARGIN
        near_bottom = y_img > height - self.FIRE_EDGE_MARGIN

        x_centered = abs(x_error) <= self.FIRE_X_CENTER_BAND
        y_centered = abs(y_error) <= self.FIRE_Y_CENTER_BAND

        # 1. If fully centered, hold for a few cycles, then drop water.
        if x_centered and y_centered:
            self.centered_fire_steps += 1
            yaw_cmd = 0.0
            pitch_cmd = 0.0

            if self.centered_fire_steps >= self.FIRE_CENTER_HOLD_STEPS:
                self.water_to_drop = 15
                self.img_coord_fire = []
                self.centered_fire_steps = 0
                print(
                    f"[{self.getName()}] Water dropped on centered fire target "
                    f"at drone position {self.current_pose[0:2]}"
                )
                self.set_fire_mode("DROP_WATER", x_img, y_img, yaw_cmd, pitch_cmd)
            else:
                self.set_fire_mode("HOLD_CENTER", x_img, y_img, yaw_cmd, pitch_cmd)

            return yaw_cmd, pitch_cmd

        self.centered_fire_steps = 0

        # 2. New override: once X is centered, run CENTER_Y before edge recovery.
        # This is the key experiment. Even if Y is near the top/bottom edge,
        # CENTER_Y wins as long as X is centered.
        if x_centered and not y_centered:
            yaw_cmd = self.yaw_towards_smoke_center(x_error, self.FIRE_SMALL_YAW)
            pitch_cmd = self.FIRE_CENTER_Y_FORCED_PITCH

            self.set_fire_mode("CENTER_Y", x_img, y_img, yaw_cmd, pitch_cmd)
            self.log_center_y_step(
                x_img,
                y_img,
                x_error,
                y_error,
                yaw_cmd,
                pitch_cmd,
                width,
                height,
                near_left,
                near_right,
                near_top,
                near_bottom,
            )
            return yaw_cmd, pitch_cmd

        # 3. Edge recovery now only handles cases where X is not centered.
        # This still protects against left/right loss before yaw alignment.
        if near_left or near_right or near_top or near_bottom:
            yaw_cmd = self.yaw_towards_smoke_center(x_error, self.FIRE_SMALL_YAW)

            if near_bottom:
                pitch_cmd = self.FIRE_SLOW_FORWARD
            elif near_top:
                pitch_cmd = self.FIRE_SLOW_BACKWARD
            else:
                pitch_cmd = 0.0

            self.set_fire_mode("EDGE_RECOVERY", x_img, y_img, yaw_cmd, pitch_cmd)
            return yaw_cmd, pitch_cmd

        # 4. If X is not centered and the target is not at an edge, yaw first.
        if not x_centered:
            yaw_cmd = self.yaw_towards_smoke_center(x_error, self.FIRE_MAX_YAW)
            pitch_cmd = 0.0

            self.set_fire_mode("ALIGN_YAW", x_img, y_img, yaw_cmd, pitch_cmd)
            return yaw_cmd, pitch_cmd

        # 5. Fallback hold.
        yaw_cmd = 0.0
        pitch_cmd = 0.0
        self.set_fire_mode("HOLD_CENTER", x_img, y_img, yaw_cmd, pitch_cmd)
        return yaw_cmd, pitch_cmd

    def should_recover_lost_fire(self):
        """
        Returns True when smoke was seen recently but is temporarily missing.

        This prevents the drone from immediately returning to waypoint patrol
        after one missed detection frame.
        """
        return (self.getTime() - self.last_fire_seen_time) <= self.FIRE_LOST_RECOVERY_SECONDS

    def recover_lost_fire(self):
        """
        Short recovery behavior when the smoke target disappears.

        The drone uses the last known image error:
            - yaw gently in the previous correction direction
            - if the smoke was near top/bottom, add a small pitch correction

        After FIRE_LOST_RECOVERY_SECONDS, the drone returns to normal patrol.
        """
        yaw_cmd = self.yaw_towards_smoke_center(self.last_fire_x_error, self.FIRE_LOST_YAW)

        if self.last_fire_y_error > self.FIRE_Y_CENTER_BAND:
            # Last seen low in the image: move forward to recover it.
            pitch_cmd = -self.FIRE_LOST_PITCH
        elif self.last_fire_y_error < -self.FIRE_Y_CENTER_BAND:
            # Last seen high in the image: move backward to recover it.
            pitch_cmd = self.FIRE_LOST_PITCH
        else:
            pitch_cmd = 0.0

        self.set_fire_mode("RECOVER_LOST_FIRE")
        return yaw_cmd, pitch_cmd

    # -------------------------------------------------------------------------
    # Fire/smoke detection.
    # -------------------------------------------------------------------------
    def fire_detection(self, verbose=False):
        """
        Detect smoke and return its image coordinate.

        Returns:
            list:
                [x, y] for the largest valid smoke blob, or [] if no reliable
                smoke target is found.

        Important:
            This method is intentionally safe. The old version could try to
            return coord_fire even when no contour radius passed the threshold.
        """
        img = self.get_image_from_camera()
        if img is None:
            return []

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Range of smoke in HSV.
        # The Sassafras Smoke proto renders as a mid-brightness grey (V ~90-145,
        # low saturation), not bright white. The previous value floor of 168
        # filtered the smoke out entirely, so it was never detected even when a
        # drone flew directly overhead. Lower the floor to catch mid-grey smoke
        # while the saturation ceiling still rejects the saturated foliage/terrain.
        smoke_lower = np.array([0, 0, 90])
        smoke_upper = np.array([172, 100, 255])

        mask_fire = cv2.inRange(hsv, smoke_lower, smoke_upper)

        fire_ratio = np.round(
            (cv2.countNonZero(mask_fire)) / (img.size / 3) * 100,
            2,
        )

        if fire_ratio <= 0.15:
            return []

        contours, _ = cv2.findContours(
            image=mask_fire,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_NONE,
        )

        best_center = None
        best_radius = 0
        contours_poly = []

        for contour in contours:
            poly = cv2.approxPolyDP(contour, 3, True)
            contours_poly.append(poly)

            center, radius = cv2.minEnclosingCircle(poly)

            if radius > 3 and radius > best_radius:
                best_center = center
                best_radius = radius

        if best_center is None:
            return []

        raw_x = float(best_center[0])
        raw_y = float(best_center[1])

        # New coordinate experiment:
        # OpenCV reports raw coordinates as (x, y), but the control logic will
        # interpret them as (control_x, control_y) = (raw_y, raw_x).
        control_x = raw_y
        control_y = raw_x
        self.last_fire_raw_coord = [raw_x, raw_y]

        if verbose:
            print(
                f"[{self.getName()}] fire detected raw=({raw_x:.2f}, {raw_y:.2f}) "
                f"control=({control_x:.2f}, {control_y:.2f})"
            )

        if self.SAVE_FIRE_DETECTION_IMAGE:
            drawing = img.copy()
            for poly in contours_poly:
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                cv2.drawContours(drawing, [poly], -1, color, 2)

            cv2.circle(
                drawing,
                (int(best_center[0]), int(best_center[1])),
                int(best_radius),
                (255, 255, 255),
                2,
            )
            cv2.imwrite(f"fire_detection_{self.getName().replace(' ', '_')}.jpg", drawing)

        return [control_x, control_y]

    # -------------------------------------------------------------------------
    # Main control loop.
    # -------------------------------------------------------------------------
    def run(self):
        t_motion = self.getTime()
        t_detection = self.getTime()
        t_water = self.getTime()

        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0

        opt_parser = optparse.OptionParser()
        opt_parser.add_option(
            "--patrol_coords",
            "--patrol_coord",
            dest="patrol_coords",
            default="11 11, 11 21, 21 21,21 11",
            help="Specify patrol coordinates in the format: x1 y1, x2 y2, ...",
        )
        opt_parser.add_option(
            "--target_altitude",
            default=42,
            type=float,
            help="Target altitude of the robot in meters",
        )
        options, _ = opt_parser.parse_args()

        point_list = options.patrol_coords.split(",")
        waypoints = []
        for point in point_list:
            values = point.split()
            if len(values) != 2:
                raise ValueError(
                    f"Invalid patrol coordinate '{point}'. "
                    "Expected format: x y, x y, ..."
                )
            waypoints.append([float(values[0]), float(values[1])])

        target_altitude = options.target_altitude

        while self.step(self.time_step) != -1:
            current_time = self.getTime()

            # Read sensors.
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            Xpos, Ypos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            self.set_position([Xpos, Ypos, altitude, roll, pitch, yaw])

            # Drop water when requested by smoke_tracking_control().
            if self.water_to_drop > 0:
                self.WaterDropStatus = True
                self.setCustomData(str(self.water_to_drop))
                self.water_to_drop = 0
            else:
                self.setCustomData(str(0))

            if altitude > target_altitude - 1:
                # Faster visual updates during smoke tracking/recovery.
                if self.img_coord_fire or self.should_recover_lost_fire():
                    detection_interval = self.FIRE_DETECTION_INTERVAL_TRACKING
                else:
                    detection_interval = self.FIRE_DETECTION_INTERVAL_PATROL

                # Fire detection happens before motion so the drone reacts to
                # fresh image coordinates instead of one-second-old coordinates.
                if current_time - t_detection > detection_interval:
                    if not self.WaterDropStatus:
                        previous_visible = self.fire_target_visible
                        detected_fire = self.fire_detection(verbose=False)

                        if detected_fire:
                            self.img_coord_fire = detected_fire
                            self.fire_target_visible = True
                            self.last_fire_seen_time = current_time

                            if not previous_visible:
                                raw_text = "None"
                                if self.last_fire_raw_coord and len(self.last_fire_raw_coord) >= 2:
                                    raw_text = (
                                        f"({self.last_fire_raw_coord[0]:.1f}, "
                                        f"{self.last_fire_raw_coord[1]:.1f})"
                                    )
                                print(
                                    f"[{self.getName()}] smoke target acquired | "
                                    f"raw_smoke={raw_text} | "
                                    f"control_smoke=({detected_fire[0]:.1f}, {detected_fire[1]:.1f})"
                                )
                        else:
                            self.img_coord_fire = []
                            self.fire_target_visible = False

                            if previous_visible:
                                print(
                                    f"[{self.getName()}] smoke target temporarily lost; "
                                    "recovering from last seen position"
                                )

                    t_detection = current_time

                # Motion update.
                if current_time - t_motion > 0.1:
                    if self.img_coord_fire:
                        yaw_disturbance, pitch_disturbance = self.smoke_tracking_control()
                    elif self.should_recover_lost_fire():
                        yaw_disturbance, pitch_disturbance = self.recover_lost_fire()
                    else:
                        self.centered_fire_steps = 0
                        self.set_fire_mode("PATROL")
                        yaw_disturbance, pitch_disturbance = self.move_to_target(waypoints)

                    t_motion = current_time

                # Wait before looking for smoke again after dropping water.
                if not self.WaterDropStatus:
                    t_water = current_time

                if current_time - t_water > 15:
                    self.WaterDropStatus = False

            roll_input = (
                self.K_ROLL_P * clamp(roll, -1, 1)
                + roll_acceleration
                + roll_disturbance
            )
            pitch_input = (
                self.K_PITCH_P * clamp(pitch, -1, 1)
                + pitch_acceleration
                + pitch_disturbance
            )
            yaw_input = yaw_disturbance

            clamped_difference_altitude = clamp( target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1, )
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            front_left_motor_input = (
                self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            )
            front_right_motor_input = (
                self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            )
            rear_left_motor_input = (
                self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            )
            rear_right_motor_input = (
                self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input
            )

            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)


if __name__ == "__main__":
    robot = Mavic()
    robot.run()
