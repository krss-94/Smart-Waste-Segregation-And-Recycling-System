"""
sensor_fusion.py
================
Simulated sensor-fusion module for the Smart Waste Segregation System.

Sensors simulated:
  - Moisture sensor  (0–100 %)
  - Gas sensor       (MQ-135 proxy, ppm equivalent)
  - Ultrasonic       (fill-level 0–100 %)  — per bin
  - Load cell        (weight in grams)

Fusion logic:
  - Each sensor generates an evidence weight for each waste category.
  - Vision confidence is multiplied by a sensor agreement multiplier.
  - Final category is determined by the highest fused score.

Author : Smart Waste Segregation System
License: MIT
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SensorReadings:
    """Container for one snapshot of all sensor values."""
    moisture_pct: float       # 0–100 %
    gas_ppm: float            # 0–1000 ppm (MQ-135 equivalent)
    weight_g: float           # grams
    timestamp: float = field(default_factory=time.time)

    # Bin fill levels (ultrasonic, per bin)
    fill_wet_pct: float = 0.0
    fill_recyclable_pct: float = 0.0
    fill_hazardous_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def any_bin_full(self) -> bool:
        return any(
            v >= 90.0
            for v in [self.fill_wet_pct, self.fill_recyclable_pct, self.fill_hazardous_pct]
        )


@dataclass
class FusionResult:
    """Output of the sensor-fusion pipeline."""
    final_category: str
    final_confidence: float
    vision_category: str
    vision_confidence: float
    sensor_boost: float          # multiplicative factor applied
    fused_scores: dict           # {"Wet": .., "Recyclable": .., "Hazardous": ..}
    sensor_readings: SensorReadings
    reasoning: list[str]         # human-readable explanation bullets

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sensor_readings"] = self.sensor_readings.to_dict()
        return d


# ---------------------------------------------------------------------------
# Sensor simulation
# ---------------------------------------------------------------------------


class SensorSimulator:
    """
    Generates realistic sensor readings given a known or inferred waste category.

    In production, replace `read()` with hardware I²C / ADC / GPIO calls.
    """

    # Typical ranges per category (mean, std)
    _PROFILES: dict[str, dict] = {
        "Wet": {
            "moisture": (75.0, 10.0),      # high moisture
            "gas": (320.0, 80.0),          # organic decomposition gases
            "weight": (250.0, 100.0),      # moderate weight
        },
        "Recyclable": {
            "moisture": (15.0, 8.0),       # low moisture (dry)
            "gas": (50.0, 20.0),           # minimal gas
            "weight": (120.0, 60.0),       # lighter items
        },
        "Hazardous": {
            "moisture": (5.0, 3.0),        # very dry
            "gas": (600.0, 150.0),         # chemical off-gassing
            "weight": (350.0, 120.0),      # batteries / e-waste heavy
        },
    }

    def __init__(
        self,
        noise_level: float = 0.1,
        bin_fill_state: Optional[dict[str, float]] = None,
        serial_port: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        noise_level     : fraction of random noise added (0–1)
        bin_fill_state  : initial fill levels {"Wet": %, "Recyclable": %, "Hazardous": %}
        serial_port     : if set, try reading from Arduino/ESP32 serial (hardware mode)
        """
        self.noise_level = noise_level
        self.bin_fill: dict[str, float] = bin_fill_state or {
            "Wet": random.uniform(10, 40),
            "Recyclable": random.uniform(5, 35),
            "Hazardous": random.uniform(0, 20),
        }
        self.serial_port = serial_port
        self._serial_conn = None
        self._item_count: int = 0

        if serial_port:
            self._init_serial(serial_port)

    def _init_serial(self, port: str) -> None:
        try:
            import serial
            self._serial_conn = serial.Serial(port, baudrate=9600, timeout=1)
            logger.info("Serial connection established on %s", port)
        except Exception as exc:
            logger.warning("Serial unavailable (%s). Using simulation.", exc)
            self._serial_conn = None

    def _clamp(self, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _noisy(self, mean: float, std: float) -> float:
        val = random.gauss(mean, std)
        noise = val * self.noise_level * random.uniform(-1, 1)
        return val + noise

    def _update_fill(self, category: str, weight_g: float) -> None:
        """Increment the correct bin's fill level based on deposited item weight."""
        # Assume a 10-litre bin (~5 kg capacity)
        increment = (weight_g / 5000.0) * 100.0
        self.bin_fill[category] = self._clamp(
            self.bin_fill[category] + increment, 0.0, 100.0
        )

    def read(self, expected_category: Optional[str] = None) -> SensorReadings:
        """
        Return a SensorReadings snapshot.

        Parameters
        ----------
        expected_category : hint for simulating realistic values.
                            If None, a random profile is chosen.
        """
        if self._serial_conn:
            return self._read_hardware()
        return self._read_simulated(expected_category)

    def _read_simulated(self, category: Optional[str]) -> SensorReadings:
        profile_key = category if category in self._PROFILES else random.choice(
            list(self._PROFILES.keys())
        )
        profile = self._PROFILES[profile_key]

        moisture = self._clamp(self._noisy(*profile["moisture"]), 0.0, 100.0)
        gas = self._clamp(self._noisy(*profile["gas"]), 0.0, 1000.0)
        weight = self._clamp(self._noisy(*profile["weight"]), 0.0, 2000.0)

        self._update_fill(profile_key, weight)
        self._item_count += 1

        return SensorReadings(
            moisture_pct=round(moisture, 2),
            gas_ppm=round(gas, 2),
            weight_g=round(weight, 2),
            fill_wet_pct=round(self.bin_fill["Wet"], 2),
            fill_recyclable_pct=round(self.bin_fill["Recyclable"], 2),
            fill_hazardous_pct=round(self.bin_fill["Hazardous"], 2),
        )

    def _read_hardware(self) -> SensorReadings:
        """
        Read from real ESP32/Arduino over serial.
        Expected serial format (CSV): moisture,gas,weight,fill_wet,fill_rec,fill_haz
        """
        try:
            line = self._serial_conn.readline().decode("utf-8").strip()
            parts = [float(x) for x in line.split(",")]
            return SensorReadings(
                moisture_pct=parts[0],
                gas_ppm=parts[1],
                weight_g=parts[2],
                fill_wet_pct=parts[3],
                fill_recyclable_pct=parts[4],
                fill_hazardous_pct=parts[5],
            )
        except Exception as exc:
            logger.warning("Hardware read failed (%s). Falling back to simulation.", exc)
            return self._read_simulated(None)

    def get_fill_status(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self.bin_fill.items()}

    def reset_bin(self, category: str) -> None:
        """Mark a bin as emptied (e.g. after collection)."""
        self.bin_fill[category] = 0.0
        logger.info("Bin '%s' has been reset to 0%%.", category)


# ---------------------------------------------------------------------------
# Fusion Engine
# ---------------------------------------------------------------------------

class SensorFusionEngine:
    """
    Combines vision-based classification with sensor evidence to produce
    a robust, fused waste-category decision.

    Fusion formula
    --------------
    fused_score[C] = vision_prob[C] * vision_weight
                   + moisture_evidence[C]
                   + gas_evidence[C]
                   + weight_evidence[C]

    The category with the highest fused score wins.
    """

    VISION_WEIGHT: float = 0.60     # vision dominates
    SENSOR_WEIGHT: float = 0.40     # sensors provide corroborating evidence

    # Moisture thresholds
    HIGH_MOISTURE_THRESHOLD: float = 60.0   # strongly suggests Wet
    LOW_MOISTURE_THRESHOLD: float = 20.0    # supports Recyclable / Hazardous

    # Gas thresholds (ppm)
    HIGH_GAS_THRESHOLD: float = 400.0       # organic decomposition or chemicals
    CHEMICAL_GAS_THRESHOLD: float = 700.0   # likely Hazardous chemicals

    def __init__(self, simulator: Optional[SensorSimulator] = None) -> None:
        self.simulator = simulator or SensorSimulator()

    def _moisture_evidence(self, moisture_pct: float) -> dict[str, float]:
        """Return evidence weights (0-1) per category based on moisture."""
        if moisture_pct >= self.HIGH_MOISTURE_THRESHOLD:
            return {"Wet": 0.85, "Recyclable": 0.05, "Hazardous": 0.10}
        elif moisture_pct <= self.LOW_MOISTURE_THRESHOLD:
            return {"Wet": 0.05, "Recyclable": 0.50, "Hazardous": 0.45}
        else:
            # Moderate moisture → slight Wet lean
            wet_score = 0.20 + 0.30 * (moisture_pct - 20.0) / 40.0
            return {"Wet": wet_score, "Recyclable": 0.50 - wet_score * 0.3, "Hazardous": 0.30}

    def _gas_evidence(self, gas_ppm: float) -> dict[str, float]:
        """Return evidence weights per category based on gas level."""
        if gas_ppm >= self.CHEMICAL_GAS_THRESHOLD:
            return {"Wet": 0.10, "Recyclable": 0.05, "Hazardous": 0.85}
        elif gas_ppm >= self.HIGH_GAS_THRESHOLD:
            return {"Wet": 0.60, "Recyclable": 0.15, "Hazardous": 0.25}
        else:
            return {"Wet": 0.20, "Recyclable": 0.65, "Hazardous": 0.15}

    def _weight_evidence(self, weight_g: float) -> dict[str, float]:
        """
        Heavier items (batteries, e-waste) lean Hazardous.
        Light dry items lean Recyclable.
        Medium-heavy wet organics lean Wet.
        """
        if weight_g > 400:
            return {"Wet": 0.20, "Recyclable": 0.25, "Hazardous": 0.55}
        elif weight_g > 150:
            return {"Wet": 0.50, "Recyclable": 0.35, "Hazardous": 0.15}
        else:
            return {"Wet": 0.20, "Recyclable": 0.65, "Hazardous": 0.15}

    def fuse(
        self,
        vision_result: dict,
        sensor_readings: Optional[SensorReadings] = None,
    ) -> FusionResult:
        """
        Fuse vision classification with sensor readings.

        Parameters
        ----------
        vision_result    : output dict from WasteClassifier.classify()
        sensor_readings  : SensorReadings (if None, simulator is called)

        Returns
        -------
        FusionResult
        """
        # ── Get sensor data ──────────────────────────────────────────
        if sensor_readings is None:
            sensor_readings = self.simulator.read(
                expected_category=vision_result.get("waste_category")
            )

        vision_cat = vision_result["waste_category"]
        vision_conf = vision_result["confidence"]

        # ── Build vision probability vector ─────────────────────────
        cat_scores_vision = vision_result.get(
            "category_scores",
            {
                vision_cat: vision_conf,
                **{
                    c: (1.0 - vision_conf) / 2
                    for c in ["Wet", "Recyclable", "Hazardous"]
                    if c != vision_cat
                },
            },
        )

        # ── Sensor evidence ──────────────────────────────────────────
        m_ev = self._moisture_evidence(sensor_readings.moisture_pct)
        g_ev = self._gas_evidence(sensor_readings.gas_ppm)
        w_ev = self._weight_evidence(sensor_readings.weight_g)

        sensor_ev: dict[str, float] = {}
        for cat in ["Wet", "Recyclable", "Hazardous"]:
            sensor_ev[cat] = (m_ev[cat] + g_ev[cat] + w_ev[cat]) / 3.0

        # ── Fused score ──────────────────────────────────────────────
        fused_scores: dict[str, float] = {}
        for cat in ["Wet", "Recyclable", "Hazardous"]:
            fused_scores[cat] = (
                self.VISION_WEIGHT * cat_scores_vision.get(cat, 0.0)
                + self.SENSOR_WEIGHT * sensor_ev[cat]
            )

        total = sum(fused_scores.values()) or 1.0
        fused_norm = {k: round(v / total, 4) for k, v in fused_scores.items()}

        final_cat = max(fused_norm, key=fused_norm.__getitem__)  # type: ignore
        final_conf = fused_norm[final_cat]

        # ── Build reasoning ──────────────────────────────────────────
        reasoning = self._build_reasoning(
            vision_cat, vision_conf, sensor_readings, m_ev, g_ev, w_ev, final_cat
        )

        # Sensor boost = how much sensor changed confidence
        sensor_boost = round(final_conf - vision_conf, 4)

        return FusionResult(
            final_category=final_cat,
            final_confidence=final_conf,
            vision_category=vision_cat,
            vision_confidence=vision_conf,
            sensor_boost=sensor_boost,
            fused_scores=fused_norm,
            sensor_readings=sensor_readings,
            reasoning=reasoning,
        )

    @staticmethod
    def _build_reasoning(
        vision_cat: str,
        vision_conf: float,
        readings: SensorReadings,
        m_ev: dict,
        g_ev: dict,
        w_ev: dict,
        final_cat: str,
    ) -> list[str]:
        reasons: list[str] = []
        reasons.append(
            f"Vision model predicted '{vision_cat}' with {vision_conf:.1%} confidence."
        )
        # Moisture
        if readings.moisture_pct >= 60:
            reasons.append(
                f"High moisture ({readings.moisture_pct:.1f}%) strongly indicates organic/wet waste."
            )
        elif readings.moisture_pct <= 20:
            reasons.append(
                f"Low moisture ({readings.moisture_pct:.1f}%) consistent with dry recyclables or hazardous items."
            )
        # Gas
        if readings.gas_ppm >= 700:
            reasons.append(
                f"Elevated gas reading ({readings.gas_ppm:.0f} ppm) suggests chemical off-gassing → Hazardous."
            )
        elif readings.gas_ppm >= 400:
            reasons.append(
                f"Moderate gas reading ({readings.gas_ppm:.0f} ppm) suggests organic decomposition → Wet."
            )
        # Weight
        if readings.weight_g > 400:
            reasons.append(
                f"High weight ({readings.weight_g:.0f} g) consistent with batteries/e-waste."
            )
        # Override notice
        if final_cat != vision_cat:
            reasons.append(
                f"⚠ Sensor fusion overrode vision: final category changed to '{final_cat}'."
            )
        else:
            reasons.append(
                f"✔ Sensor readings corroborate vision classification: '{final_cat}'."
            )
        return reasons


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sim = SensorSimulator()
    engine = SensorFusionEngine(simulator=sim)

    mock_vision = {
        "waste_category": "Wet",
        "confidence": 0.72,
        "category_scores": {"Wet": 0.72, "Recyclable": 0.18, "Hazardous": 0.10},
    }

    result = engine.fuse(mock_vision)
    print("\n=== Sensor Fusion Result ===")
    print(f"  Vision      : {result.vision_category} ({result.vision_confidence:.1%})")
    print(f"  Final       : {result.final_category} ({result.final_confidence:.1%})")
    print(f"  Sensor boost: {result.sensor_boost:+.4f}")
    print(f"  Readings    : moisture={result.sensor_readings.moisture_pct}%  "
          f"gas={result.sensor_readings.gas_ppm} ppm  "
          f"weight={result.sensor_readings.weight_g} g")
    print("\n  Reasoning:")
    for r in result.reasoning:
        print(f"    • {r}")
    print("\n  Bin Fill Levels:")
    print(f"    Wet        : {result.sensor_readings.fill_wet_pct:.1f}%")
    print(f"    Recyclable : {result.sensor_readings.fill_recyclable_pct:.1f}%")
    print(f"    Hazardous  : {result.sensor_readings.fill_hazardous_pct:.1f}%")
