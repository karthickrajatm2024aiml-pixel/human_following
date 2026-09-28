import time
import serial

PORT = "/dev/ttyUSB0"
BAUD = 115200

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=0.2,
    write_timeout=1
)

print("ESP32 connected")
time.sleep(3)

def send(left, right):
    command = f"L:{left},R:{right}\n"
    ser.write(command.encode())
    ser.flush()
    print("[PI]", command.strip())

try:
    print("\nFORWARD TEST")
    start = time.time()

    while time.time() - start < 3:
        send(150, 150)

        while ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print("[ESP32]", line)

        time.sleep(0.2)

    print("\nSTOP")
    send(0, 0)
    time.sleep(1)

finally:
    send(0, 0)
    ser.close()
    print("Test finished")
