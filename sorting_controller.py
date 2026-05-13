"""
sorting_controller.py
=====================
Servo-based sorting controller for the Smart Waste Segregation System.

Servo angle mapping:
    Wet Waste        →   0°   (left bin)
    Recyclable Waste →  90°   (centre bin)
    Hazardous Waste  → 180°   (right bin)

Supports:
  - Simulated servo output (default — no hardware required)
  - Real servo control via Arduino/ESP32 serial (optional)
  - PWM duty-cycle calculation for direct GPIO (Raspberry Pi / ESP32)

Author : Smart Waste Segregation System
License: MIT
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVO_ANGLES: dict[str, int] = {
    "Wet": 0,
    "Recyclable": 90,
    "Hazardous": 180,
}

SERVO_RETURN_ANGLE: int = 90    # neutral position after sorting
SERVO_SETTLE_MS: int = 500      # wait time for servo to reach position (ms)

# PWM parameters for SG90 / MG996R servos (50 Hz, 1-2 ms pulse)
PWM_FREQUENCY: int = 50
PWM_MIN_DUTY: float = 2.5      # duty cycle at 0°
PWM_MAX_DUTY: float = 12.5     # duty cycle at 180°


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SortingAction:
    """Record of a single sorting event."""
    waste_category: str
    servo_angle: int
    pwm_duty_cycle: float
    timestamp: float = field(default_factory=time.time)
    simulated: bool = True
    success: bool = True
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SortingStats:
    """Running statistics for the sorting system."""
    total_sorted: int = 0
    wet_count: int = 0
    recyclable_count: int = 0
    hazardous_count: int = 0
    errors: int = 0

    def increment(self, category: str) -> None:
        self.total_sorted += 1
        if category == "Wet":
            self.wet_count += 1
        elif category == "Recyclable":
            self.recyclable_count += 1
        elif category == "Hazardous":
            self.hazardous_count += 1

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Servo Controller
# ---------------------------------------------------------------------------


class ServoController:
    """
    Controls a servo motor to physically sort waste into the correct bin.

    Parameters
    ----------
    serial_port  : COM port / /dev/ttyUSB0 for Arduino/ESP32 serial control.
                   If None, runs in simulation mode.
    gpio_pin     : GPIO pin number for direct PWM (Raspberry Pi / ESP32 GPIO).
                   Only used when pyRPi.GPIO is available.
    simulate     : Force simulation mode regardless of hardware.
    """

    def __init__(
        self,
        serial_port: Optional[str] = None,
        gpio_pin: Optional[int] = None,
        simulate: bool = True,
    ) -> None:
        self.simulate = simulate
        self.serial_port = serial_port
        self.gpio_pin = gpio_pin
        self._serial_conn = None
        self._gpio_servo = None
        self._current_angle: int = SERVO_RETURN_ANGLE
        self.stats = SortingStats()
        self._history: list[SortingAction] = []

        if not simulate:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Hardware init
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        """Try to initialise serial or GPIO hardware."""
        if self.serial_port:
            self._init_serial()
        elif self.gpio_pin is not None:
            self._init_gpio()
        else:
            logger.warning("No hardware interface specified; falling back to simulation.")
            self.simulate = True

    def _init_serial(self) -> None:
        try:
            import serial
            self._serial_conn = serial.Serial(
                self.serial_port, baudrate=9600, timeout=2
            )
            time.sleep(2)  # Allow Arduino to reset
            logger.info("Serial servo control ready on %s", self.serial_port)
            self.simulate = False
        except Exception as exc:
            logger.warning("Serial init failed (%s). Using simulation.", exc)
            self.simulate = True

    def _init_gpio(self) -> None:
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT)
            self._gpio_servo = GPIO.PWM(self.gpio_pin, PWM_FREQUENCY)
            self._gpio_servo.start(self._angle_to_duty(SERVO_RETURN_ANGLE))
            logger.info("GPIO servo PWM ready on pin %d", self.gpio_pin)
            self.simulate = False
        except Exception as exc:
            logger.warning("GPIO init failed (%s). Using simulation.", exc)
            self.simulate = True

    # ------------------------------------------------------------------
    # Angle ↔ duty cycle
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_to_duty(angle: int) -> float:
        """Convert servo angle (0-180°) to PWM duty cycle (2.5–12.5%)."""
        return PWM_MIN_DUTY + (angle / 180.0) * (PWM_MAX_DUTY - PWM_MIN_DUTY)

    # ------------------------------------------------------------------
    # Movement commands
    # ------------------------------------------------------------------

    def move_to(self, angle: int) -> bool:
        """
        Move servo to the specified angle.

        Returns True on success, False on failure.
        """
        angle = max(0, min(180, angle))
        if self.simulate:
            return self._move_simulated(angle)
        elif self._serial_conn:
            return self._move_serial(angle)
        elif self._gpio_servo:
            return self._move_gpio(angle)
        else:
            return self._move_simulated(angle)

    def _move_simulated(self, angle: int) -> bool:
        logger.info(
            "[SIM] Servo moving: %d° → %d°  (PWM duty: %.2f%%)",
            self._current_angle,
            angle,
            self._angle_to_duty(angle),
        )
        time.sleep(SERVO_SETTLE_MS / 1000.0)
        self._current_angle = angle
        return True

    def _move_serial(self, angle: int) -> bool:
        try:
            command = f"SERVO:{angle}\n".encode("utf-8")
            self._serial_conn.write(command)
            # Wait for ACK
            ack = self._serial_conn.readline().decode("utf-8").strip()
            if ack.startswith("OK"):
                self._current_angle = angle
                return True
            logger.warning("Unexpected serial ACK: %s", ack)
            return False
        except Exception as exc:
            logger.error("Serial move failed: %s", exc)
            return False

    def _move_gpio(self, angle: int) -> bool:
        try:
            duty = self._angle_to_duty(angle)
            self._gpio_servo.ChangeDutyCycle(duty)
            time.sleep(SERVO_SETTLE_MS / 1000.0)
            self._current_angle = angle
            return True
        except Exception as exc:
            logger.error("GPIO move failed: %s", exc)
            return False

    def return_to_neutral(self) -> bool:
        """Move servo back to the neutral/centre position."""
        return self.move_to(SERVO_RETURN_ANGLE)

    # ------------------------------------------------------------------
    # Sorting logic
    # ------------------------------------------------------------------

    def sort(self, waste_category: str) -> SortingAction:
        """
        Execute a complete sort cycle for the given waste category.

        Sequence:
            1. Look up servo angle for category.
            2. Move to sorting angle.
            3. Brief hold (item drops into bin).
            4. Return to neutral.
            5. Record action and update stats.

        Parameters
        ----------
        waste_category : "Wet" | "Recyclable" | "Hazardous"

        Returns
        -------
        SortingAction dataclass
        """
        if waste_category not in SERVO_ANGLES:
            logger.warning("Unknown category '%s'; defaulting to Recyclable.", waste_category)
            waste_category = "Recyclable"

        angle = SERVO_ANGLES[waste_category]
        duty = self._angle_to_duty(angle)

        logger.info(
            "Sorting '%s' → servo angle %d° (duty %.2f%%)",
            waste_category, angle, duty
        )

        # Execute movement
        success = self.move_to(angle)
        time.sleep(0.8)           # Hold — item falls into bin
        self.return_to_neutral()

        # Record
        action = SortingAction(
            waste_category=waste_category,
            servo_angle=angle,
            pwm_duty_cycle=round(duty, 3),
            simulated=self.simulate,
            success=success,
            message=(
                f"Successfully sorted into {waste_category} bin."
                if success
                else "Sort failed — servo did not respond."
            ),
        )

        if success:
            self.stats.increment(waste_category)
        else:
            self.stats.errors += 1

        self._history.append(action)
        return action

    # ------------------------------------------------------------------
    # Status & history
    # ------------------------------------------------------------------

    @property
    def current_angle(self) -> int:
        return self._current_angle

    def get_history(self, last_n: int = 20) -> list[dict]:
        """Return the last N sorting actions as dicts."""
        return [a.to_dict() for a in self._history[-last_n:]]

    def get_status(self) -> dict:
        return {
            "current_angle": self._current_angle,
            "simulate": self.simulate,
            "stats": self.stats.to_dict(),
            "angle_map": SERVO_ANGLES,
        }

    def cleanup(self) -> None:
        """Release hardware resources."""
        if self._gpio_servo:
            self._gpio_servo.stop()
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
            except Exception:
                pass
        if self._serial_conn and self._serial_conn.is_open:
            self._serial_conn.close()
        logger.info("ServoController cleaned up.")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Visual indicator (ANSI terminal)
# ---------------------------------------------------------------------------

CATEGORY_COLOURS = {
    "Wet":        "\033[92m",   # green
    "Recyclable": "\033[94m",   # blue
    "Hazardous":  "\033[91m",   # red
}
RESET = "\033[0m"

BIN_ART = {
    "Wet":        "🟢 [ WET BIN        0° ] ←←←",
    "Recyclable": "🔵 [ RECYCLE BIN   90° ] ↓↓↓",
    "Hazardous":  "🔴 [ HAZARD BIN   180° ] →→→",
}


def print_sort_result(action: SortingAction) -> None:
    colour = CATEGORY_COLOURS.get(action.waste_category, "")
    art = BIN_ART.get(action.waste_category, "?")
    print(f"\n  {colour}{art}{RESET}")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │ Category   : {action.waste_category:<27} │")
    print(f"  │ Angle      : {action.servo_angle}°{'':<26} │")
    print(f"  │ PWM Duty   : {action.pwm_duty_cycle:.3f}%{'':<24} │")
    print(f"  │ Mode       : {'Simulated' if action.simulated else 'Hardware':<27} │")
    print(f"  │ Status     : {'✔ OK' if action.success else '✖ FAILED':<27} │")
    print(f"  └─────────────────────────────────────────┘")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ctrl = ServoController(simulate=True)
    for cat in ["Wet", "Recyclable", "Hazardous", "Recyclable"]:
        print(f"\n→ Sorting: {cat}")
        action = ctrl.sort(cat)
        print_sort_result(action)

    print("\n=== Sorting Stats ===")
    for k, v in ctrl.stats.to_dict().items():
        print(f"  {k}: {v}")
