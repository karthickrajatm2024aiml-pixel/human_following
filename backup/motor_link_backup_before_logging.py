#!/usr/bin/env python3
"""
motor_link.py

Thin serial link from the Pi to the ESP32 differential-drive controller
(esp32_diff_drive.ino). The ESP32 is execution-only -- it never decides
where to go, it just does what this link tells it and stops on its own if
the Pi goes quiet. All steering logic (SteeringController) lives on the
Pi side, in web_dashboard.py.

Protocol (ASCII, newline-terminated, 115200 baud):
    Pi -> ESP32:  "L:<int>,R:<int>\n"   signed PWM, -255..255, per side
                  "S\n"                  explicit stop (same as L:0,R:0)
                  "PING\n"                liveness check
    ESP32 -> Pi:  "READY\n"              once, on boot
                  "PONG\n"               reply to PING

The ESP32 independently zeroes both motors if no valid L/R or S command
arrives within WATCHDOG_TIMEOUT_MS (500ms, see the .ino). This class does
NOT need to actively "keepalive" beyond sending steering commands at a
normal rate -- if the Pi process dies or the serial cable is unplugged,
the ESP32 stops on its own without this class doing anything further.
"""
import time

try:
    import serial
except ImportError:
    serial = None


class MotorLink:
    """Not thread-safe on its own -- caller (FrameWorker) already keeps all
    pipeline/motor access on a single thread, so no extra locking is added
    here to match that existing convention."""

    def __init__(self, port, baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.connected = False

        if not port:
            print("[motor] no --drive-port given -- running vision-only, "
                  "motors will never be commanded.", flush=True)
            return

        if serial is None:
            print("[motor] pyserial not installed (pip install pyserial) -- "
                  "driving disabled, dashboard will still run in vision-only mode.", flush=True)
            return

        try:
            self.ser = serial.Serial(port, baud, timeout=0.05)
            # Opening the port toggles DTR on most ESP32 boards, which
            # resets the chip -- give it time to boot and print READY
            # before commands start flowing.
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            self.connected = True
            print(f"[motor] connected to ESP32 on {port} @ {baud}", flush=True)
        except Exception as e:
            print(f"[motor] could not open {port}: {e} -- driving disabled, "
                  f"dashboard will still run in vision-only mode.", flush=True)
            self.ser = None
            self.connected = False

    def send_drive(self, left, right):
        left = int(max(-255, min(255, left)))
        right = int(max(-255, min(255, right)))
        self._write(f"L:{left},R:{right}\n")

    def stop(self):
        self._write("S\n")

    def _write(self, line):
        if not self.connected or self.ser is None:
            return
        try:
            self.ser.write(line.encode("ascii"))
        except Exception as e:
            print(f"[motor] write failed ({e}) -- marking link down until restart", flush=True)
            self.connected = False

    def close(self):
        if self.ser is not None:
            try:
                self.stop()
                time.sleep(0.05)
                self.ser.close()
            except Exception:
                pass
