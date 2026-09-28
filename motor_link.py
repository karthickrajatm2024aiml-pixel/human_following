#!/usr/bin/env python3

"""
motor_link.py

Serial communication between Raspberry Pi and ESP32
for the ShopMate autonomous human-following cart.

Protocol
--------

Raspberry Pi -> ESP32

    L:<left>,R:<right>
    S
    PING

Examples:

    L:120,R:120
    L:-55,R:55
    L:80,R:120
    S
    PING


ESP32 -> Raspberry Pi

    READY
    PONG
    DRIVE L:120 R:120
    STOP
    WATCHDOG
    ERR ...


This version:
- logs Pi -> ESP32 commands
- logs ESP32 -> Pi replies
- performs a startup serial flush
- clears partial garbage in the ESP32 command buffer
"""

import threading
import time

try:
    import serial
except ImportError:
    serial = None


class MotorLink:

    def __init__(
        self,
        port=None,
        baud=115200,
        timeout=0.05
    ):
        self.port = port
        self.baud = int(baud)
        self.timeout = float(timeout)

        self.ser = None
        self.connected = False

        self._write_lock = threading.Lock()
        self._stop_reader = threading.Event()
        self._reader_thread = None

        self.last_left = 0
        self.last_right = 0

        # ------------------------------------------------------
        # No motor port requested
        # ------------------------------------------------------

        if not self.port:
            print(
                "[motor] no drive port specified - motor output disabled",
                flush=True
            )
            return

        # ------------------------------------------------------
        # pyserial check
        # ------------------------------------------------------

        if serial is None:
            print(
                "[motor] ERROR: pyserial is not installed",
                flush=True
            )
            return

        # ------------------------------------------------------
        # Open serial connection
        # ------------------------------------------------------

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                write_timeout=0.5
            )

            # ESP32 may reset when serial port is opened.
            time.sleep(3.0)

            # Clear anything already waiting on Raspberry Pi side.
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass

            # IMPORTANT:
            # Send a blank command terminator.
            #
            # This clears any partial startup garbage sitting inside
            # the ESP32 serialCommand buffer.
            try:
                self.ser.write(b"\n")
                self.ser.flush()
            except Exception:
                pass

            time.sleep(0.3)

            # Clear any READY / ERR / startup replies generated
            # before normal communication begins.
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass

            self.connected = True

            print(
                f"[motor] connected to ESP32 on "
                f"{self.port} @ {self.baud}",
                flush=True
            )

            print(
                "[motor] startup serial buffer flushed",
                flush=True
            )

            # Start background reader so ESP32 replies can be
            # displayed while web_dashboard_v2.py is running.
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                daemon=True
            )

            self._reader_thread.start()

        except Exception as exc:
            self.ser = None
            self.connected = False

            print(
                f"[motor] ERROR opening {self.port}: {exc}",
                flush=True
            )

    # =========================================================
    # Utility
    # =========================================================

    @staticmethod
    def _clamp_pwm(value):

        try:
            value = int(round(float(value)))
        except Exception:
            value = 0

        if value > 255:
            value = 255

        if value < -255:
            value = -255

        return value

    # =========================================================
    # Background ESP32 reader
    # =========================================================

    def _reader_loop(self):

        buffer = bytearray()

        while not self._stop_reader.is_set():

            if (
                not self.connected
                or
                self.ser is None
            ):
                time.sleep(0.05)
                continue

            try:
                waiting = self.ser.in_waiting

                if waiting > 0:

                    data = self.ser.read(waiting)

                    if data:
                        buffer.extend(data)

                    while b"\n" in buffer:

                        line_bytes, _, remainder = (
                            buffer.partition(b"\n")
                        )

                        buffer = bytearray(remainder)

                        line = (
                            line_bytes
                            .decode(
                                "utf-8",
                                errors="ignore"
                            )
                            .replace("\r", "")
                            .strip()
                        )

                        if line:
                            print(
                                f"[ESP32 -> PI] {line}",
                                flush=True
                            )

                else:
                    time.sleep(0.01)

            except Exception as exc:

                if not self._stop_reader.is_set():

                    print(
                        f"[motor] serial read error: {exc}",
                        flush=True
                    )

                self.connected = False
                break

    # =========================================================
    # Send raw command
    # =========================================================

    def _send_line(
        self,
        command,
        print_command=True
    ):

        if (
            not self.connected
            or
            self.ser is None
        ):
            return False

        command = str(command).strip()

        if not command:
            return False

        payload = (
            command + "\n"
        ).encode("utf-8")

        try:

            with self._write_lock:

                self.ser.write(payload)
                self.ser.flush()

            if print_command:

                print(
                    f"[PI -> ESP32] {command}",
                    flush=True
                )

            return True

        except Exception as exc:

            print(
                f"[motor] serial write error: {exc}",
                flush=True
            )

            self.connected = False

            return False

    # =========================================================
    # Drive
    # =========================================================

    def send_drive(
        self,
        left,
        right
    ):

        left = self._clamp_pwm(left)
        right = self._clamp_pwm(right)

        self.last_left = left
        self.last_right = right

        print(
            f"[MOTOR] L={left} R={right}",
            flush=True
        )

        command = (
            f"L:{left},R:{right}"
        )

        return self._send_line(command)


    # =========================================================
    # Stop
    # =========================================================

    def stop(self):

        self.last_left = 0
        self.last_right = 0

        return self._send_line("STOP")

    # =========================================================
    # Ping
    # =========================================================

    def ping(self):

        return self._send_line("PING")

    # =========================================================
    # Close
    # =========================================================

    def close(self):

        # Try to stop the motors before disconnecting.
        if (
            self.connected
            and
            self.ser is not None
        ):
            try:
                self.stop()
                time.sleep(0.05)
            except Exception:
                pass

        self._stop_reader.set()

        if (
            self._reader_thread is not None
            and
            self._reader_thread.is_alive()
        ):
            try:
                self._reader_thread.join(
                    timeout=0.5
                )
            except Exception:
                pass

        if self.ser is not None:

            try:
                self.ser.close()
            except Exception:
                pass

        self.ser = None
        self.connected = False

        print(
            "[motor] serial connection closed",
            flush=True
        )


# =============================================================
# Direct test
# =============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0"
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200
    )

    args = parser.parse_args()

    motor = MotorLink(
        args.port,
        baud=args.baud
    )

    if not motor.connected:

        raise SystemExit(
            "ESP32 serial connection failed."
        )

    try:

        print(
            "\nTesting PING...",
            flush=True
        )

        motor.ping()

        time.sleep(0.7)

        print(
            "\nTesting FORWARD...",
            flush=True
        )

        motor.send_drive(
            120,
            120
        )

        time.sleep(0.5)

        print(
            "\nTesting LEFT...",
            flush=True
        )

        motor.send_drive(
            -120,
            120
        )

        time.sleep(0.5)

        print(
            "\nTesting RIGHT...",
            flush=True
        )

        motor.send_drive(
            120,
            -120
        )

        time.sleep(0.5)

        print(
            "\nTesting STOP...",
            flush=True
        )

        motor.stop()

        time.sleep(0.5)

    finally:

        motor.close()
