# sleepy guard app
# real-time driver monitoring using face landmarks and adaptive scoring

import time

from .utils import beep_desktop, is_raspberry_pi


# class that groups related logic/state for this component
class AlarmController:
# initialize class state and configuration
    def __init__(self, enable_alarm: bool, gpio_pin: int = 18):
        self.enable_alarm = enable_alarm
        self.gpio_pin = gpio_pin
        self.pi_mode = False
        self.gpio = None

        if not enable_alarm:
            return

        if is_raspberry_pi():
            try:
                import RPi.GPIO as GPIO  # type: ignore

                self.gpio = GPIO
                self.gpio.setwarnings(False)
                self.gpio.setmode(GPIO.BCM)
                self.gpio.setup(self.gpio_pin, GPIO.OUT)
                self.gpio.output(self.gpio_pin, GPIO.LOW)
                self.pi_mode = True
            except Exception:
                self.pi_mode = False

# function that handles a specific step in the pipeline
    def trigger(self, duration_s: float = 0.15):
        if not self.enable_alarm:
            return
        if self.pi_mode and self.gpio is not None:
            try:
                self.gpio.output(self.gpio_pin, self.gpio.HIGH)
                time.sleep(duration_s)
                self.gpio.output(self.gpio_pin, self.gpio.LOW)
                return
            except Exception:
                pass
        beep_desktop()

# function that handles a specific step in the pipeline
    def cleanup(self):
        if self.pi_mode and self.gpio is not None:
            try:
                self.gpio.output(self.gpio_pin, self.gpio.LOW)
                self.gpio.cleanup(self.gpio_pin)
            except Exception:
                pass
