#!/usr/bin/env python3

"""
web_dashboard.py

Raspberry Pi human-following dashboard.

Pipeline:

    Camera
       |
       v
    YOLOv8n
       |
       v
    IoU/Centroid Tracker
       |
       v
    Target
       |
       v
    Filtered steering controller
       |
       v
    MotorLink
       |
       v
    ESP32
       |
       v
    L298N

Driving is OFF by default.

No obstacle sensing is used in this version.
"""

import argparse
import asyncio
import base64
import json
import queue
import sys
import threading
import time

import cv2
import numpy as np
from aiohttp import web, WSMsgType

from live_follow_smoke_test import (
    build_detect_fn,
    build_embed_fn,
    IMGSZ,
)

from tracker_reid import Pipeline
from motor_link import MotorLink


# ============================================================
# DISPLAY
# ============================================================

TARGET_COLOR = (0, 220, 0)
TRACK_COLOR = (0, 200, 255)

CLICK_MISS_RADIUS_PX = 80.0


# ============================================================
# PERFORMANCE
# ============================================================

DEFAULT_PROCESS_FPS = 8.0
DEFAULT_STREAM_FPS = 12.0


# ============================================================
# FOLLOWING PARAMETERS
# ============================================================

# Desired apparent person height.
#
# Increase this -> robot follows closer.
# Decrease this -> robot follows farther away.
#
TARGET_BOX_HEIGHT_FRAC = 0.50


# Distance deadband.
#
# Prevents constant forward/backward movement.
HEIGHT_DEADBAND = 0.07


# Horizontal deadband.
#
# Prevents twitching when person is nearly centered.
TURN_DEADBAND = 0.08


# Controller gains.
FORWARD_GAIN = 260.0
TURN_GAIN = 145.0


# Motor limits.
MAX_FORWARD_PWM = 120
MAX_REVERSE_PWM = 75
MAX_TURN_PWM = 105


# Your physical steering direction.
#
# We already established that the original direction was reversed.
TURN_SIGN = -1


# Minimum useful PWM.
#
# L298N + motors may not move at very low PWM.
MIN_MOVE_PWM = 50


# ============================================================
# FILTERING
# ============================================================

# Target center smoothing.
#
# Higher = faster response
# Lower  = smoother
POSITION_ALPHA = 0.35


# Target size smoothing.
SIZE_ALPHA = 0.30


# Motor smoothing.
MOTOR_ALPHA = 0.30


# Maximum PWM change per processed frame.
#
# This is extremely important for preventing sudden jerks.
MAX_PWM_STEP = 22


# ============================================================
# TARGET VALIDATION
# ============================================================

# Ignore very small people.
MIN_TARGET_HEIGHT_FRAC = 0.10


# Don't react to a target that jumps unrealistically
# between two frames.
MAX_POSITION_JUMP_FRAC = 0.40


# ============================================================
# SEARCH
# ============================================================

SEARCH_PWM = 55

# Approximate time for one full search rotation.
# This must be calibrated on the real cart because there is no encoder/IMU feedback.
SEARCH_360_TIME_S = 6.0

# If the customer dashboard is not connected when the customer is lost,
# keep the cart stopped and wait up to this long for the phone to reconnect.
CUSTOMER_RECONNECT_WAIT_S = 30.0


class SteeringController:
    """
    Smooth human-following controller.

    Uses:
        - filtered target X position
        - filtered target size
        - proportional control
        - nonlinear turning
        - motor smoothing
        - PWM slew limiting
        - reduced forward speed while turning
    """

    def __init__(self):

        self.last_turn_sign = 1

        self.lost_since = None
        self.search_completed = False

        # Filtered target position.
        self.filtered_cx = None
        self.filtered_height = None

        # Last motor command.
        self.prev_left = 0.0
        self.prev_right = 0.0

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    # --------------------------------------------------------
    # Exponential filter
    # --------------------------------------------------------

    @staticmethod
    def ema(previous, current, alpha):

        if previous is None:
            return current

        return (
            alpha * current
            +
            (1.0 - alpha) * previous
        )

    # --------------------------------------------------------
    # PWM slew limiter
    # --------------------------------------------------------

    def slew(self, previous, target):

        delta = target - previous

        if delta > MAX_PWM_STEP:
            delta = MAX_PWM_STEP

        elif delta < -MAX_PWM_STEP:
            delta = -MAX_PWM_STEP

        return previous + delta

    # --------------------------------------------------------
    # Motor smoothing
    # --------------------------------------------------------

    def smooth_motor(self, previous, target):

        return (
            previous * (1.0 - MOTOR_ALPHA)
            +
            target * MOTOR_ALPHA
        )

    # --------------------------------------------------------
    # Differential mixing
    # --------------------------------------------------------

    @staticmethod
    def mix(fwd, turn):

        turn *= TURN_SIGN

        left = fwd + turn
        right = fwd - turn

        left = max(-255, min(255, left))
        right = max(-255, min(255, right))

        return left, right

    # --------------------------------------------------------
    # Stop smoothly
    # --------------------------------------------------------

    def smooth_stop(self):

        self.prev_left = self.slew(
            self.prev_left,
            0
        )

        self.prev_right = self.slew(
            self.prev_right,
            0
        )

        if abs(self.prev_left) < 5:
            self.prev_left = 0

        if abs(self.prev_right) < 5:
            self.prev_right = 0

        return (
            int(self.prev_left),
            int(self.prev_right)
        )

    # --------------------------------------------------------
    # Main controller
    # --------------------------------------------------------

    def compute(
        self,
        frame_w,
        frame_h,
        result,
        target_enrolled
    ):

        # ----------------------------------------------------
        # No target
        # ----------------------------------------------------

        if not target_enrolled:

            self.lost_since = None
            self.search_completed = False

            self.filtered_cx = None
            self.filtered_height = None

            self.prev_left = 0
            self.prev_right = 0

            return 0, 0, "no-target"

        # ----------------------------------------------------
        # Locate target
        # ----------------------------------------------------

        target = next(
            (
                t
                for t in result.tracks
                if t.is_target
            ),
            None
        )

        # ----------------------------------------------------
        # Target lost
        # ----------------------------------------------------

        if target is None or result.target_lost:

            now = time.time()

            if self.lost_since is None:
                self.lost_since = now
                self.search_completed = False

            elapsed = now - self.lost_since

            # Search finished: stay stopped until the customer is found again
            # or a new target is enrolled.
            if self.search_completed or elapsed >= SEARCH_360_TIME_S:

                self.search_completed = True
                left, right = self.smooth_stop()

                return (
                    left,
                    right,
                    "customer-lost-after-360-search"
                )

            # Approximate 360-degree search rotation.
            # The camera keeps running during this state, so if the target
            # reappears, normal following resumes immediately.
            turn = (
                self.last_turn_sign
                *
                SEARCH_PWM
            )

            left, right = self.mix(
                0,
                turn
            )

            left = self.smooth_motor(
                self.prev_left,
                left
            )

            right = self.smooth_motor(
                self.prev_right,
                right
            )

            left = self.slew(
                self.prev_left,
                left
            )

            right = self.slew(
                self.prev_right,
                right
            )

            self.prev_left = left
            self.prev_right = right

            return (
                int(left),
                int(right),
                "searching-360"
            )

        # ----------------------------------------------------
        # Target visible
        # ----------------------------------------------------

        self.lost_since = None
        self.search_completed = False

        x1, y1, x2, y2 = target.bbox

        cx = (
            x1 + x2
        ) / 2.0

        box_h = max(
            1.0,
            y2 - y1
        )

        height_frac = (
            box_h
            /
            float(frame_h)
        )

        # ----------------------------------------------------
        # Reject extremely small detections
        # ----------------------------------------------------

        if height_frac < MIN_TARGET_HEIGHT_FRAC:

            left, right = self.smooth_stop()

            return (
                left,
                right,
                "target-too-small"
            )

        # ----------------------------------------------------
        # Target position filter
        # ----------------------------------------------------

        if self.filtered_cx is not None:

            jump = abs(
                cx - self.filtered_cx
            )

            jump_frac = (
                jump
                /
                float(frame_w)
            )

            # Ignore one-frame giant detector jumps.
            if jump_frac > MAX_POSITION_JUMP_FRAC:

                cx = self.filtered_cx

        self.filtered_cx = self.ema(
            self.filtered_cx,
            cx,
            POSITION_ALPHA
        )

        # ----------------------------------------------------
        # Target size filter
        # ----------------------------------------------------

        self.filtered_height = self.ema(
            self.filtered_height,
            height_frac,
            SIZE_ALPHA
        )

        # ----------------------------------------------------
        # Horizontal error
        # ----------------------------------------------------

        turn_error = (
            self.filtered_cx
            -
            frame_w / 2.0
        ) / (
            frame_w / 2.0
        )

        turn_error = self.clamp(
            turn_error,
            -1.0,
            1.0
        )

        # ----------------------------------------------------
        # Remember search direction
        # ----------------------------------------------------

        if abs(turn_error) > TURN_DEADBAND:

            self.last_turn_sign = (
                1
                if turn_error > 0
                else -1
            )

        # ----------------------------------------------------
        # TURN CONTROL
        # ----------------------------------------------------

        if abs(turn_error) <= TURN_DEADBAND:

            turn_pwm = 0.0

        else:

            # Normalize error after deadband.
            effective_error = (
                abs(turn_error)
                -
                TURN_DEADBAND
            ) / (
                1.0
                -
                TURN_DEADBAND
            )

            effective_error = self.clamp(
                effective_error,
                0.0,
                1.0
            )

            # Quadratic response:
            #
            # small error -> gentle
            # large error -> stronger

            magnitude = (
                effective_error
                *
                effective_error
                *
                TURN_GAIN
            )

            turn_pwm = (
                magnitude
                if turn_error > 0
                else -magnitude
            )

            turn_pwm = self.clamp(
                turn_pwm,
                -MAX_TURN_PWM,
                MAX_TURN_PWM
            )

        # ----------------------------------------------------
        # DISTANCE CONTROL
        # ----------------------------------------------------

        height_error = (
            TARGET_BOX_HEIGHT_FRAC
            -
            self.filtered_height
        )

        if abs(height_error) <= HEIGHT_DEADBAND:

            forward_pwm = 0.0

        else:

            forward_pwm = (
                height_error
                *
                FORWARD_GAIN
            )

            forward_pwm = self.clamp(
                forward_pwm,
                -MAX_REVERSE_PWM,
                MAX_FORWARD_PWM
            )

        # ----------------------------------------------------
        # Slow down when turning.
        #
        # If person is far left/right:
        # turn first, don't drive forward hard.
        # ----------------------------------------------------

        turn_ratio = (
            abs(turn_pwm)
            /
            float(MAX_TURN_PWM)
        )

        turn_ratio = self.clamp(
            turn_ratio,
            0.0,
            1.0
        )

        forward_pwm *= (
            1.0
            -
            0.70 * turn_ratio
        )

        # ----------------------------------------------------
        # Minimum motor PWM
        # ----------------------------------------------------

        if (
            abs(forward_pwm) > 0
            and
            abs(forward_pwm) < MIN_MOVE_PWM
        ):

            forward_pwm = (
                MIN_MOVE_PWM
                if forward_pwm > 0
                else -MIN_MOVE_PWM
            )

        if (
            abs(turn_pwm) > 0
            and
            abs(turn_pwm) < MIN_MOVE_PWM
        ):

            turn_pwm = (
                MIN_MOVE_PWM
                if turn_pwm > 0
                else -MIN_MOVE_PWM
            )

        # ----------------------------------------------------
        # Differential drive mix
        # ----------------------------------------------------

        target_left, target_right = self.mix(
            forward_pwm,
            turn_pwm
        )

        # ----------------------------------------------------
        # Motor EMA
        # ----------------------------------------------------

        left = self.smooth_motor(
            self.prev_left,
            target_left
        )

        right = self.smooth_motor(
            self.prev_right,
            target_right
        )

        # ----------------------------------------------------
        # PWM slew limiting
        # ----------------------------------------------------

        left = self.slew(
            self.prev_left,
            left
        )

        right = self.slew(
            self.prev_right,
            right
        )

        self.prev_left = left
        self.prev_right = right

        # ----------------------------------------------------
        # Near zero -> STOP
        # ----------------------------------------------------

        if (
            abs(left) < 8
            and
            abs(right) < 8
        ):

            self.prev_left = 0
            self.prev_right = 0

            return (
                0,
                0,
                "following-stop"
            )

        return (
            int(left),
            int(right),
            "following"
        )


# ============================================================
# FRAME WORKER
# ============================================================

class FrameWorker(threading.Thread):

    def __init__(
        self,
        cap,
        pipeline,
        clients,
        motor_link,
        jpeg_quality=70,
        max_fps=12.0,
        process_fps=8.0
    ):

        super().__init__(
            daemon=True
        )

        self.cap = cap
        self.pipeline = pipeline
        self.clients = clients
        self.motor_link = motor_link

        self.jpeg_quality = jpeg_quality

        self.max_fps = max_fps
        self.process_fps = process_fps

        self.min_frame_interval = (
            1.0 / max_fps
            if max_fps > 0
            else 0
        )

        self.process_interval = (
            1.0 / process_fps
            if process_fps > 0
            else 0
        )

        self.commands = queue.Queue()

        self.loop = None

        self._stop_event = threading.Event()

        self.drive_enabled = False

        self.steering = SteeringController()

        self._latest_frame = None
        self._latest_tracks = []

        self._latest_drive_state = "no-target"

        # Customer-lost notification state.
        self._customer_lost_event_started = None
        self._customer_lost_notified = False
        self._customer_lost_wait_expired = False

    # --------------------------------------------------------
    # Thread-safe loop attach
    # --------------------------------------------------------

    def attach_loop(self, loop):

        self.loop = loop

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------

    def stop(self):

        self._stop_event.set()

        self.drive_enabled = False

        if self.motor_link is not None:

            try:
                self.motor_link.stop()
            except Exception:
                pass

    # --------------------------------------------------------
    # Command queue
    # --------------------------------------------------------

    def enqueue(self, command):

        self.commands.put(command)

    # --------------------------------------------------------
    # Commands from browser
    # --------------------------------------------------------

    def _drain_cmds(self):

        while True:

            try:
                cmd = self.commands.get_nowait()

            except queue.Empty:

                break

            self._handle_command(cmd)

    # --------------------------------------------------------
    # Command handler
    # --------------------------------------------------------

    def _handle_command(self, cmd):

        name = cmd.get("cmd")

        # ----------------------------------------------------
        # ENABLE
        # ----------------------------------------------------

        if name == "drive_enable":

            self.drive_enabled = True

            print(
                "[drive] ENABLED",
                flush=True
            )

            return

        # ----------------------------------------------------
        # DISABLE
        # ----------------------------------------------------

        if name == "drive_disable":

            self.drive_enabled = False

            if self.motor_link:

                self.motor_link.stop()

            print(
                "[drive] DISABLED",
                flush=True
            )

            return

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        if name == "stop":

            self.drive_enabled = False

            if self.motor_link:

                self.motor_link.stop()

            self.steering.prev_left = 0
            self.steering.prev_right = 0

            print(
                "[drive] EMERGENCY STOP",
                flush=True
            )

            return

        # ----------------------------------------------------
        # ENROLL NEAREST
        # ----------------------------------------------------

        if name == "enroll":

            self._enroll_nearest()

            return

        # ----------------------------------------------------
        # ENROLL AT CLICK
        # ----------------------------------------------------

        if name == "enroll_at":

            try:

                x = float(cmd["x"])
                y = float(cmd["y"])

                self._enroll_at(
                    x,
                    y
                )

            except Exception as e:

                print(
                    f"[enroll] invalid click: {e}",
                    flush=True
                )

            return

    # --------------------------------------------------------
    # Nearest target
    # --------------------------------------------------------

    def _enroll_nearest(self):

        if self._latest_frame is None:

            print(
                "[enroll] no frame yet",
                flush=True
            )

            return

        tracks = self._latest_tracks

        if not tracks:

            print(
                "[enroll] no people detected",
                flush=True
            )

            return

        frame_h, frame_w = (
            self._latest_frame.shape[:2]
        )

        center_x = frame_w / 2.0
        center_y = frame_h / 2.0

        best = None
        best_distance = None

        for track in tracks:

            x1, y1, x2, y2 = track.bbox

            cx = (
                x1 + x2
            ) / 2.0

            cy = (
                y1 + y2
            ) / 2.0

            distance = (
                (cx - center_x) ** 2
                +
                (cy - center_y) ** 2
            )

            if (
                best_distance is None
                or
                distance < best_distance
            ):

                best = track
                best_distance = distance

        if best is not None:

            self.pipeline.enroll_target(
                self._latest_frame,
                best.bbox
            )

            print(
                f"[enroll] nearest target "
                f"id={best.track_id}",
                flush=True
            )

            self.steering.filtered_cx = None
            self.steering.filtered_height = None

    # --------------------------------------------------------
    # Click target
    # --------------------------------------------------------

    def _enroll_at(self, x, y):

        if self._latest_frame is None:

            print(
                "[enroll] no frame yet",
                flush=True
            )

            return

        best = None
        best_distance = None

        for track in self._latest_tracks:

            x1, y1, x2, y2 = track.bbox

            dx = max(
                x1 - x,
                0.0,
                x - x2
            )

            dy = max(
                y1 - y,
                0.0,
                y - y2
            )

            distance = (
                dx * dx
                +
                dy * dy
            ) ** 0.5

            if (
                best_distance is None
                or
                distance < best_distance
            ):

                best = track
                best_distance = distance

        if (
            best is None
            or
            best_distance > CLICK_MISS_RADIUS_PX
        ):

            print(
                "[enroll] click did not hit a person",
                flush=True
            )

            return

        self.pipeline.enroll_target(
            self._latest_frame,
            best.bbox
        )

        print(
            f"[enroll] target locked "
            f"id={best.track_id}",
            flush=True
        )

        self.steering.filtered_cx = None
        self.steering.filtered_height = None

    # --------------------------------------------------------
    # Customer-lost notification handling
    # --------------------------------------------------------

    def _handle_customer_lost_notification(self, drive_state):

        lost_state = (
            drive_state == "customer-lost-after-360-search"
        )

        if not lost_state:
            # Reset only after the loss event has cleared, so a future
            # loss can generate a new notification.
            self._customer_lost_event_started = None
            self._customer_lost_notified = False
            self._customer_lost_wait_expired = False
            return

        now = time.time()

        if self._customer_lost_event_started is None:
            self._customer_lost_event_started = now

        elapsed = now - self._customer_lost_event_started

        # If the customer's phone/dashboard is connected, notify immediately.
        if self.clients and not self._customer_lost_notified:

            self._broadcast({
                "type": "notification",
                "kind": "customer_lost",
                "title": "ShopMate: Customer Lost",
                "message": (
                    "Customer could not be found after the 360-degree search. "
                    "The cart has stopped."
                )
            })

            self._customer_lost_notified = True

            print(
                "[notify] customer-lost notification sent",
                flush=True
            )

            return

        # No phone/dashboard connection: wait 30 seconds for reconnection.
        if (
            not self.clients
            and not self._customer_lost_wait_expired
            and elapsed >= CUSTOMER_RECONNECT_WAIT_S
        ):

            self._customer_lost_wait_expired = True

            print(
                f"[notify] no customer connection after "
                f"{CUSTOMER_RECONNECT_WAIT_S:.0f}s; cart remains stopped",
                flush=True
            )

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    def _draw(
        self,
        frame,
        result,
        fps,
        drive_state
    ):

        h, w = frame.shape[:2]

        # Center line.
        cv2.line(
            frame,
            (w // 2, 0),
            (w // 2, h),
            (255, 255, 255),
            1
        )

        for track in result.tracks:

            color = (
                TARGET_COLOR
                if track.is_target
                else TRACK_COLOR
            )

            x1, y1, x2, y2 = (
                int(v)
                for v in track.bbox
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = (
                f"id={track.track_id}"
                +
                (
                    " TARGET"
                    if track.is_target
                    else ""
                )
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(0, y1 - 6)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        status = (
            f"fps={fps:.1f} "
            f"tracks={len(result.tracks)} "
            f"drive="
            f"{'ON' if self.drive_enabled else 'OFF'}:"
            f"{drive_state}"
        )

        if result.target_lost:

            status += " TARGET LOST"

        cv2.putText(
            frame,
            status,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        return frame

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    def run(self):

        last_stream_send = 0.0
        last_process = 0.0

        last_fps_t = time.time()

        frame_count = 0

        fps = 0.0

        print(
            f"[process] vision capped at "
            f"{self.process_fps:.1f} FPS",
            flush=True
        )

        print(
            f"[stream] browser capped at "
            f"{self.max_fps:.1f} FPS",
            flush=True
        )

        while not self._stop_event.is_set():

            # ----------------------------------------------
            # Browser commands
            # ----------------------------------------------

            self._drain_cmds()

            # ----------------------------------------------
            # Read camera
            # ----------------------------------------------

            ok, frame = self.cap.read()

            if not ok:

                print(
                    "[worker] camera read failed",
                    flush=True
                )

                break

            now = time.time()

            # ----------------------------------------------
            # Process only at configured FPS
            # ----------------------------------------------

            if (
                last_process == 0.0
                or
                now - last_process
                >= self.process_interval
            ):

                last_process = now

                try:

                    result = (
                        self.pipeline
                        .process_frame_raw(frame)
                    )

                except Exception as e:

                    print(
                        f"[worker] pipeline error: {e}",
                        flush=True
                    )

                    break

                self._latest_frame = frame.copy()

                self._latest_tracks = (
                    result.tracks
                )

                # ------------------------------------------
                # Steering
                # ------------------------------------------

                frame_h, frame_w = (
                    frame.shape[:2]
                )

                target_enrolled = (
                    self.pipeline
                    .target_embedding
                    is not None
                )

                # ------------------------------------------
                # Autonomous-follow state
                #
                # IMPORTANT:
                # When driving is disabled, do NOT run follow,
                # 360-degree search, or lost-customer notification
                # logic. Keep the cart stopped and reset any
                # previous search/loss state.
                # ------------------------------------------

                if self.drive_enabled:

                    left, right, drive_state = (
                        self.steering.compute(
                            frame_w,
                            frame_h,
                            result,
                            target_enrolled
                        )
                    )

                    self._handle_customer_lost_notification(
                        drive_state
                    )

                else:

                    left = 0
                    right = 0
                    drive_state = "disabled"

                    # Reset follow/search state so enabling driving
                    # later starts cleanly instead of continuing an
                    # old lost-customer timer.
                    self.steering.lost_since = None
                    self.steering.search_completed = False
                    self.steering.prev_left = 0
                    self.steering.prev_right = 0

                    # Reset notification state while drive is OFF.
                    self._customer_lost_event_started = None
                    self._customer_lost_notified = False
                    self._customer_lost_wait_expired = False

                self._latest_drive_state = (
                    drive_state
                )

                # ------------------------------------------
                # Motor command
                # ------------------------------------------

                if (
                    self.drive_enabled
                    and
                    self.motor_link is not None
                ):

                    self.motor_link.send_drive(
                        left,
                        right
                    )

                frame_count += 1

                if (
                    now - last_fps_t
                    >= 1.0
                ):

                    fps = (
                        frame_count
                        /
                        (
                            now - last_fps_t
                        )
                    )

                    frame_count = 0

                    last_fps_t = now

                # ------------------------------------------
                # Draw latest processed result
                # ------------------------------------------

                annotated = self._draw(
                    frame.copy(),
                    result,
                    fps,
                    drive_state
                )

                self._latest_frame = annotated

            else:

                # No inference this frame.
                # Keep camera capture alive.
                continue

            # ----------------------------------------------
            # Browser stream rate
            # ----------------------------------------------

            now = time.time()

            if (
                now - last_stream_send
                <
                self.min_frame_interval
            ):

                continue

            last_stream_send = now

            if self._latest_frame is None:

                continue

            ok, buf = cv2.imencode(
                ".jpg",
                self._latest_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    self.jpeg_quality
                ]
            )

            if not ok:

                continue

            target_enrolled = (
                self.pipeline.target_embedding
                is not None
            )

            payload = {

                "type": "frame",

                "image":
                    base64.b64encode(
                        buf
                    ).decode("ascii"),

                "n_tracks":
                    len(self._latest_tracks),

                "target_id":
                    getattr(
                        result,
                        "target_track_id",
                        None
                    ),

                "target_lost":
                    getattr(
                        result,
                        "target_lost",
                        False
                    ),

                "target_enrolled":
                    target_enrolled,

                "fps":
                    round(fps, 1),

                "drive_enabled":
                    self.drive_enabled,

                "drive_state":
                    self._latest_drive_state,

                "drive_connected":
                    bool(
                        self.motor_link
                        and
                        self.motor_link.connected
                    ),
            }

            self._broadcast(payload)

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        if self.motor_link is not None:

            try:
                self.motor_link.stop()
            except Exception:
                pass

            try:
                self.motor_link.close()
            except Exception:
                pass

        self.cap.release()

        print(
            "[worker] stopped",
            flush=True
        )

    # --------------------------------------------------------
    # Broadcast
    # --------------------------------------------------------

    def _broadcast(self, payload):

        if self.loop is None:
            return

        message = json.dumps(
            payload
        )

        for ws in list(
            self.clients
        ):

            asyncio.run_coroutine_threadsafe(
                self._safe_send(
                    ws,
                    message
                ),
                self.loop
            )

    @staticmethod
    async def _safe_send(
        ws,
        message
    ):

        try:

            if not ws.closed:

                await ws.send_str(
                    message
                )

        except Exception:
            pass


# ============================================================
# HTML
# ============================================================

INDEX_HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<title>Follow-Bot Dashboard</title>

<style>

body {
    background:#111;
    color:#eee;
    font-family:system-ui,sans-serif;
    margin:0;
    padding:16px;
}

#wrap {
    display:flex;
    flex-direction:column;
    align-items:center;
    gap:12px;
}

h1 {
    font-size:20px;
    margin:0;
}

#hint {
    color:#8ab4ff;
    font-size:13px;
}

#stream {
    width:720px;
    max-width:100%;
    background:#000;
    border-radius:8px;
    cursor:crosshair;
}

#status {
    display:flex;
    gap:20px;
    flex-wrap:wrap;
    color:#aaa;
}

#driveStatus {
    font-size:14px;
}

#notificationBanner {
    display:none;
    width:min(720px, 100%);
    box-sizing:border-box;
    padding:12px 14px;
    border-radius:8px;
    background:#8b1e1e;
    color:#fff;
    font-weight:700;
    text-align:center;
}

#conn {
    color:#888;
    font-size:12px;
}

.row {
    display:flex;
    gap:10px;
    flex-wrap:wrap;
    justify-content:center;
}

button {
    border:0;
    border-radius:6px;
    padding:10px 18px;
    background:#2b6ef2;
    color:white;
    cursor:pointer;
    font-size:14px;
}

button:hover {
    background:#1e57c4;
}

#stopBtn {
    background:#c62828;
    font-weight:bold;
}

#driveToggleBtn.enabled {
    background:#4caf50;
}

.ok {
    color:#4caf50;
}

.warn {
    color:#ff5555;
}

</style>

</head>

<body>

<div id="wrap">

<h1>Follow-Bot Dashboard</h1>

<div id="hint">
Click on a person to select the target.
</div>

<img
    id="stream"
    alt="waiting for camera..."
>

<div id="status">

<span id="fps">
fps: --
</span>

<span id="tracks">
tracks: --
</span>

<span id="target">
target: --
</span>

</div>

<div id="driveStatus">
drive: OFF
</div>

<div id="notificationBanner"></div>

<div class="row">

<button id="lockBtn">
Lock Nearest to Center
</button>

<button id="driveToggleBtn">
Enable Driving
</button>

<button id="stopBtn">
STOP
</button>

</div>

<div id="clickMsg"></div>

<div id="conn">
connecting...
</div>

</div>


<script>

let ws = null;
let driveEnabled = false;


function connect()
{
    ws = new WebSocket(
        `ws://${location.host}/ws`
    );


    ws.onopen = () =>
    {
        document.getElementById(
            'conn'
        ).textContent = 'connected';
    };


    ws.onclose = () =>
    {
        document.getElementById(
            'conn'
        ).textContent =
            'disconnected -- retrying...';

        setTimeout(
            connect,
            1500
        );
    };


    ws.onerror = () =>
    {
        ws.close();
    };


    ws.onmessage = (evt) =>
    {
        const data =
            JSON.parse(evt.data);

        if (data.type === 'notification')
        {
            const banner = document.getElementById(
                'notificationBanner'
            );

            banner.textContent = data.message || 'Customer lost';
            banner.style.display = 'block';

            if (
                'Notification' in window &&
                Notification.permission === 'granted'
            )
            {
                try
                {
                    new Notification(
                        data.title || 'ShopMate',
                        { body: data.message || '' }
                    );
                }
                catch (e)
                {
                    // The in-page banner still works even when the
                    // browser blocks OS notifications on local HTTP.
                }
            }

            return;
        }

        if (data.type !== 'frame')
            return;


        document.getElementById(
            'stream'
        ).src =
            'data:image/jpeg;base64,' +
            data.image;


        document.getElementById(
            'fps'
        ).textContent =
            `fps: ${data.fps}`;


        document.getElementById(
            'tracks'
        ).textContent =
            `tracks: ${data.n_tracks}`;


        const target =
            document.getElementById(
                'target'
            );


        if (!data.target_enrolled)
        {
            target.textContent =
                'target: not enrolled';

            target.className = '';
        }

        else if (data.target_lost)
        {
            target.textContent =
                'target: LOST';

            target.className =
                'warn';
        }

        else
        {
            target.textContent =
                `target: locked (id=${data.target_id})`;

            target.className =
                'ok';

            // Customer has been found again. Hide the old
            // lost-customer warning banner.
            const banner = document.getElementById(
                'notificationBanner'
            );

            banner.textContent = '';
            banner.style.display = 'none';
        }


        driveEnabled =
            !!data.drive_enabled;

        // If autonomous driving is OFF, clear any stale
        // customer-lost warning from an earlier drive session.
        if (!driveEnabled)
        {
            const banner = document.getElementById(
                'notificationBanner'
            );

            banner.textContent = '';
            banner.style.display = 'none';
        }


        const button =
            document.getElementById(
                'driveToggleBtn'
            );


        button.textContent =
            driveEnabled
                ? 'Disable Driving'
                : 'Enable Driving';


        button.classList.toggle(
            'enabled',
            driveEnabled
        );


        const status =
            document.getElementById(
                'driveStatus'
            );


        status.textContent =
            `drive: ${
                driveEnabled
                    ? 'ON'
                    : 'OFF'
            } (${data.drive_state})` +
            (
                data.drive_connected
                    ? ''
                    : ' [no ESP32 link]'
            );
    };
}


// ------------------------------------------------------------
// Click target
// ------------------------------------------------------------

document.getElementById(
    'stream'
).addEventListener(
    'click',
    (evt) =>
    {
        const img = evt.target;

        if (
            !img.naturalWidth ||
            !img.naturalHeight
        )
        {
            return;
        }


        const rect =
            img.getBoundingClientRect();


        const scaleX =
            img.naturalWidth /
            rect.width;


        const scaleY =
            img.naturalHeight /
            rect.height;


        const x =
            (evt.clientX - rect.left)
            * scaleX;


        const y =
            (evt.clientY - rect.top)
            * scaleY;


        if (
            ws &&
            ws.readyState === WebSocket.OPEN
        )
        {
            ws.send(
                JSON.stringify({
                    cmd:'enroll_at',
                    x:x,
                    y:y
                })
            );
        }
    }
);


// ------------------------------------------------------------
// Lock nearest
// ------------------------------------------------------------

document.getElementById(
    'lockBtn'
).onclick = () =>
{
    if (
        ws &&
        ws.readyState === WebSocket.OPEN
    )
    {
        ws.send(
            JSON.stringify({
                cmd:'enroll'
            })
        );
    }
};


// ------------------------------------------------------------
// Drive
// ------------------------------------------------------------

document.getElementById(
    'driveToggleBtn'
).onclick = async () =>
{
    if (
        'Notification' in window &&
        Notification.permission === 'default'
    )
    {
        try
        {
            await Notification.requestPermission();
        }
        catch (e)
        {
            // In-page notification remains available.
        }
    }

    if (
        !ws ||
        ws.readyState !== WebSocket.OPEN
    )
        return;


    ws.send(
        JSON.stringify({
            cmd:
                driveEnabled
                    ? 'drive_disable'
                    : 'drive_enable'
        })
    );
};


// ------------------------------------------------------------
// STOP
// ------------------------------------------------------------

document.getElementById(
    'stopBtn'
).onclick = () =>
{
    if (
        ws &&
        ws.readyState === WebSocket.OPEN
    )
    {
        ws.send(
            JSON.stringify({
                cmd:'stop'
            })
        );
    }
};


connect();

</script>

</body>

</html>
"""


# ============================================================
# HTTP
# ============================================================

async def index_handler(request):

    return web.Response(
        text=INDEX_HTML,
        content_type="text/html"
    )


# ============================================================
# WEBSOCKET
# ============================================================

async def ws_handler(request):

    ws = web.WebSocketResponse()

    await ws.prepare(request)

    clients = request.app["clients"]

    clients.add(ws)

    print(
        "[ws] browser connected",
        flush=True
    )

    try:

        async for msg in ws:

            if msg.type == WSMsgType.TEXT:

                try:

                    data = json.loads(
                        msg.data
                    )

                    request.app[
                        "worker"
                    ].enqueue(data)

                except Exception as e:

                    print(
                        f"[ws] command error: {e}",
                        flush=True
                    )

            elif msg.type == WSMsgType.ERROR:

                break

    finally:

        clients.discard(ws)

        print(
            "[ws] browser disconnected",
            flush=True
        )

    return ws


# ============================================================
# APP
# ============================================================

def build_app(
    cap,
    pipeline,
    motor_link,
    jpeg_quality,
    max_stream_fps,
    process_fps
):

    app = web.Application()

    app["clients"] = set()

    worker = FrameWorker(
        cap,
        pipeline,
        app["clients"],
        motor_link,
        jpeg_quality=jpeg_quality,
        max_fps=max_stream_fps,
        process_fps=process_fps
    )

    app["worker"] = worker

    app.router.add_get(
        "/",
        index_handler
    )

    app.router.add_get(
        "/ws",
        ws_handler
    )


    async def startup(app):

        worker.attach_loop(
            asyncio.get_running_loop()
        )

        worker.start()


    async def shutdown(app):

        worker.stop()

        worker.join(
            timeout=3.0
        )


    app.on_startup.append(
        startup
    )

    app.on_shutdown.append(
        shutdown
    )

    return app


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0"
    )

    parser.add_argument(
        "--yolo-onnx",
        default="yolov8n.onnx"
    )

    parser.add_argument(
        "--reid-onnx",
        default="reid_proxy.onnx"
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=65
    )

    parser.add_argument(
        "--max-stream-fps",
        type=float,
        default=DEFAULT_STREAM_FPS
    )

    parser.add_argument(
        "--process-fps",
        type=float,
        default=DEFAULT_PROCESS_FPS
    )

    parser.add_argument(
        "--drive-port",
        default=None
    )

    parser.add_argument(
        "--drive-baud",
        type=int,
        default=115200
    )

    args = parser.parse_args()


    # --------------------------------------------------------
    # ONNX Runtime
    # --------------------------------------------------------

    try:

        import onnxruntime as ort

    except ImportError:

        print(
            "[error] onnxruntime is not installed",
            flush=True
        )

        sys.exit(1)


    print(
        f"[load] {args.yolo_onnx}",
        flush=True
    )


    # Limit ONNX threads.
    #
    # This reduces unnecessary CPU saturation and heat
    # while keeping the detector responsive.

    det_options = (
        ort.SessionOptions()
    )

    det_options.intra_op_num_threads = 3
    det_options.inter_op_num_threads = 1

    det_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )


    det_sess = ort.InferenceSession(
        args.yolo_onnx,
        sess_options=det_options,
        providers=[
            "CPUExecutionProvider"
        ]
    )


    det_input_name = (
        det_sess
        .get_inputs()[0]
        .name
    )


    print(
        f"[load] {args.reid_onnx}",
        flush=True
    )


    reid_options = (
        ort.SessionOptions()
    )

    reid_options.intra_op_num_threads = 2
    reid_options.inter_op_num_threads = 1

    reid_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )


    reid_sess = ort.InferenceSession(
        args.reid_onnx,
        sess_options=reid_options,
        providers=[
            "CPUExecutionProvider"
        ]
    )


    reid_input_name = (
        reid_sess
        .get_inputs()[0]
        .name
    )


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    detect_fn = build_detect_fn(
        det_sess,
        det_input_name
    )

    embed_fn = build_embed_fn(
        reid_sess,
        reid_input_name
    )


    pipeline = Pipeline(
        embed_fn=embed_fn,
        detect_fn=detect_fn,
        match_threshold=args.match_threshold
    )


    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    src = (
        int(args.source)
        if str(args.source).isdigit()
        else args.source
    )


    cap = cv2.VideoCapture(
        src
    )


    if not cap.isOpened():

        print(
            f"[error] could not open camera: {args.source}",
            flush=True
        )

        sys.exit(1)


    # Try to reduce camera buffering.
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )


    # --------------------------------------------------------
    # Motor
    # --------------------------------------------------------

    motor_link = MotorLink(
        args.drive_port,
        baud=args.drive_baud
    )


    # --------------------------------------------------------
    # Startup information
    # --------------------------------------------------------

    print(
        f"[imgsz] detector: {IMGSZ}",
        flush=True
    )

    print(
        f"[process] process FPS: {args.process_fps}",
        flush=True
    )

    print(
        f"[stream] stream FPS: {args.max_stream_fps}",
        flush=True
    )

    print(
        "[onnx] detector threads: 3",
        flush=True
    )

    print(
        "[onnx] re-id threads: 2",
        flush=True
    )

    print(
        f"[serve] http://{args.host}:{args.port}/",
        flush=True
    )

    print(
        "[drive] starts DISABLED",
        flush=True
    )


    app = build_app(
        cap,
        pipeline,
        motor_link,
        args.jpeg_quality,
        args.max_stream_fps,
        args.process_fps
    )


    web.run_app(
        app,
        host=args.host,
        port=args.port,
        print=None
    )


if __name__ == "__main__":

    main()