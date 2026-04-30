import time

# sends and receives simple telemetry data to/from the esp8266 microcontroller over the serial connection
class SerialTelemetry:
    # default port can be changed with --serial-port, baud rate can be changed with --serial-baud, and send interval can be changed with --serial-interval
    def __init__(self, enabled=False, port="/dev/ttyUSB0", baud=115200, interval=0.5):
        self.enabled = enabled
        self.port = port
        self.baud = baud
        self.interval = interval
        self.last_send = 0.0
        self.serial = None

        if not self.enabled:
            return

        # attempt to open the serial port and connect to the esp8266
        try:
            import serial
            self.serial = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(2.0)
            print(f"[serial] connected to esp8266 on {self.port} at {self.baud} baud")
        except Exception as exc:
            self.serial = None
            print(f"[serial] could not open esp8266 serial port {self.port}: {exc}")

    # periodically send the stats to the esp
    def send(self, status, attentiveness, drowsy_score):
        if not self.enabled or self.serial is None:
            return

        now = time.time()
        if now - self.last_send < self.interval:
            return

        self.last_send = now

        try:
            # simple csv-style line to be parsed on esp8266
            msg = f"{status},{attentiveness:.1f},{drowsy_score:.3f}\n"
            self.serial.write(msg.encode("utf-8"))
            self.serial.flush()

        except Exception as exc:
            print(f"[serial] send failed: {exc}")
            try:
                self.serial.close()
            except Exception:
                pass
            self.serial = None

    # read any commands sent from the esp8266 via serial
    def read_esp_command(self):
        if self.serial is None:
            return None

        if self.serial.in_waiting <= 0:
            return None

        line = self.serial.readline().decode("utf-8", errors="ignore").strip()

        # handle the button press commands sent from the esp8266
        if line == "BTN_RESET_BASELINE":
            return "RESET_BASELINE"

        if line == "BTN_RESET_STATS":
            return "RESET_STATS"

        return None
    
    # gracefully close the serial connection
    def close(self):
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
            self.serial = None
