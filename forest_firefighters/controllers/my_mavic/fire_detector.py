"""
fire_detector.py
Perception module for the My Mavic forest firefighting controller.
Detects fire/smoke in camera frames using HSV colour segmentation and spatial filtering.
"""

import sys
sys.path.insert(0, '/usr/local/webots/lib/controller/python39')
sys.path.insert(0, '/home/tusiim3/.local/lib/python3.9/site-packages')

import numpy as np
import cv2


class FireDetector:
    """
    Detects smoke/fire in a Webots camera image using HSV segmentation.
    Enforces a strict spatial mass requirement to filter out ambient noise.
    """

    # Smoke HSV range — highly reflective achromatic elements
    SMOKE_LOWER = np.array([0, 0, 100])
    SMOKE_UPPER = np.array([179, 100, 255])

    # Minimum area in pixels for a single connected component to be deemed valid
    MIN_CONTOUR_AREA = 25.0  

    # Minimum global ratio (%) of validated fire pixels to trigger tracking state
    FIRE_RATIO_THRESHOLD = 0.05

    def __init__(self, camera_width, camera_height):
        self.width = camera_width
        self.height = camera_height

    def get_processed_image(self, camera):
        """
        Pull raw image from Webots camera device and convert to RGB numpy array.
        """
        raw = camera.getImage()
        img = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return cv2.flip(img, 1)

    def detect(self, camera):
        """
        Run fire/smoke detection on the current camera frame.

        Returns:
            detected (bool): True only if valid structural contours exceed threshold
            cx (float):      Image X coordinate of fire center (pixels)
            cy (float):      Image Y coordinate of fire center (pixels)
            radius (float):  Radius of largest valid contour (pixels)
        """
        img = self.get_processed_image(camera)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.SMOKE_LOWER, self.SMOKE_UPPER)

        # Extract discrete contours from the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, 0.0, 0.0, 0.0

        cx, cy, radius_max = 0.0, 0.0, 0.0
        total_valid_pixels = 0.0

        for c in contours:
            area = cv2.contourArea(c)
            
            # Reject scattered pixel noise and isolated reflections
            if area >= self.MIN_CONTOUR_AREA:
                total_valid_pixels += area
                
                # Compute enclosing circle bounds for tracking
                poly = cv2.approxPolyDP(c, 3, True)
                center, radius = cv2.minEnclosingCircle(poly)
                
                # Keep track of the most significant cluster
                if radius > radius_max:
                    cx, cy = center
                    radius_max = radius

        # Calculate ratio strictly from validated, high-mass pixel clusters
        total_pixels = self.width * self.height
        fire_ratio = np.round((total_valid_pixels / total_pixels) * 100, 3)

        # Confirm a structural mass has been identified and exceeds minimum density
        if radius_max > 0.0 and fire_ratio >= self.FIRE_RATIO_THRESHOLD:
            print(f"[FireDetector] Valid fire signature isolated: ratio={fire_ratio}%, max_r={radius_max:.1f}px")
            return True, cx, cy, radius_max

        # Fall through to negative detection if components are too small
        return False, 0.0, 0.0, 0.0

    def is_centered(self, cx, cy, threshold=20):
        """
        Returns True when fire centroid is within threshold pixels of frame centre.
        """
        return (abs(cx - self.width / 2) <= threshold and
                abs(cy - self.height / 2) <= threshold)
