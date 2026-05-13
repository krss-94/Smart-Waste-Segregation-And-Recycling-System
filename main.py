"""
main.py
=======
Entry point for the Smart Waste Segregation and Recycling System.

Usage
-----
  python main.py                          # classify a default sample image
  python main.py --image path/to/img.jpg  # classify a specific image
  python main.py --webcam                 # use live webcam feed
  python main.py --dashboard              # launch IoT dashboard (port 5000)
  python main.py --demo                   # run a multi-item demo cycle

Author : Smart Waste Segregation System
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is on the path when running from project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
SRC  = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Rich console (optional, falls back to plain print)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None  # type: ignore


def _print(msg: str, style: str = "") -> None:
    if HAS_RICH and console:
        console.print(msg, style=style)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

def _load_modules():
    """Lazy-import all system modules (allows --help without TF installed)."""
    from classifier import WasteClassifier
    from sensor_fusion import SensorSimulator, SensorFusionEngine
    from sorting_controller import ServoController, print_sort_result
    return WasteClassifier, SensorSimulator, SensorFusionEngine, ServoController, print_sort_result


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source,
    classifier,
    fusion_engine,
    controller,
    verbose: bool = True,
) -> dict:
    """
    Execute one full waste-classification and sorting cycle.

    Steps:
      1. Classify image (vision)
      2. Read sensor data
      3. Fuse vision + sensors
      4. Trigger servo sort
      5. Print results

    Returns the fusion result dict.
    """
    t0 = time.time()

    # ── 1. Vision classification ─────────────────────────────────────
    logger.info("Step 1/4 — Image classification …")
    vision = classifier.classify_with_fallback(source)

    # ── 2 & 3. Sensor fusion ─────────────────────────────────────────
    logger.info("Step 2/4 — Reading sensors …")
    sensors = fusion_engine.simulator.read(expected_category=vision["waste_category"])

    logger.info("Step 3/4 — Fusing vision + sensor evidence …")
    fusion = fusion_engine.fuse(vision, sensors)

    # ── 4. Servo sort ────────────────────────────────────────────────
    logger.info("Step 4/4 — Triggering servo sort …")
    action = controller.sort(fusion.final_category)

    elapsed = round(time.time() - t0, 2)

    if verbose:
        _print_results(vision, fusion, action, elapsed)

    return fusion.to_dict()


def _print_results(vision, fusion, action, elapsed: float) -> None:
    """Print a formatted summary to the terminal."""
    COLOURS = {"Wet": "green", "Recyclable": "blue", "Hazardous": "red"}
    cat = fusion.final_category
    colour = COLOURS.get(cat, "white")

    if HAS_RICH and console:
        # Rich output
        table = Table(box=box.ROUNDED, border_style="bright_black", expand=False)
        table.add_column("Field", style="dim", width=22)
        table.add_column("Value", style="bold")

        table.add_row("Final Category",   f"[{colour}]{cat}[/{colour}]")
        table.add_row("Final Confidence", f"{fusion.final_confidence:.1%}")
        table.add_row("Vision Prediction",f"{fusion.vision_category} ({fusion.vision_confidence:.1%})")
        table.add_row("Sensor Boost",     f"{fusion.sensor_boost:+.4f}")
        table.add_row("ImageNet Label",   vision.get("imagenet_label", "—"))
        table.add_row("", "")
        table.add_row("Moisture",         f"{fusion.sensor_readings.moisture_pct:.1f} %")
        table.add_row("Gas (ppm)",        f"{fusion.sensor_readings.gas_ppm:.0f}")
        table.add_row("Weight",           f"{fusion.sensor_readings.weight_g:.0f} g")
        table.add_row("", "")
        table.add_row("Servo Angle",      f"{action.servo_angle}°")
        table.add_row("PWM Duty",         f"{action.pwm_duty_cycle:.3f} %")
        table.add_row("Servo Mode",       "Simulated" if action.simulated else "Hardware")
        table.add_row("", "")
        table.add_row("Bin — Wet",        f"{fusion.sensor_readings.fill_wet_pct:.1f} %")
        table.add_row("Bin — Recyclable", f"{fusion.sensor_readings.fill_recyclable_pct:.1f} %")
        table.add_row("Bin — Hazardous",  f"{fusion.sensor_readings.fill_hazardous_pct:.1f} %")
        table.add_row("", "")
        table.add_row("Pipeline Time",    f"{elapsed} s")

        console.print(Panel(table, title="[bold]♻  Smart Waste Segregation Result[/bold]",
                            border_style=colour))

        console.print("\n[bold dim]Fusion Reasoning:[/bold dim]")
        for reason in fusion.reasoning:
            console.print(f"  • {reason}", style="dim")
    else:
        # Plain output
        print("\n" + "=" * 54)
        print("  SMART WASTE SEGREGATION — RESULT")
        print("=" * 54)
        print(f"  Final Category   : {cat}")
        print(f"  Confidence       : {fusion.final_confidence:.1%}")
        print(f"  Vision Prediction: {fusion.vision_category} ({fusion.vision_confidence:.1%})")
        print(f"  Sensor Boost     : {fusion.sensor_boost:+.4f}")
        print(f"  Servo Angle      : {action.servo_angle}°")
        print(f"  Pipeline Time    : {elapsed}s")
        print("-" * 54)
        print("  Reasoning:")
        for r in fusion.reasoning:
            print(f"    • {r}")
        print("=" * 54)

    # Servo art (always shown)
    from sorting_controller import print_sort_result
    print_sort_result(action)

    # Bin alert
    readings = fusion.sensor_readings
    for name, fill in [("Wet", readings.fill_wet_pct),
                       ("Recyclable", readings.fill_recyclable_pct),
                       ("Hazardous", readings.fill_hazardous_pct)]:
        if fill >= 80:
            _print(f"\n  ⚠  [{name}] bin is {fill:.0f}% full — please empty soon!", style="yellow")


# ---------------------------------------------------------------------------
# Webcam mode
# ---------------------------------------------------------------------------

def run_webcam(classifier, fusion_engine, controller) -> None:
    """Continuously classify frames from the default webcam (press Q to quit)."""
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Run: pip install opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        sys.exit(1)

    _print("[bold green]Webcam mode started. Press 'c' to classify, 'q' to quit.[/bold green]")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Smart Waste — Press C to classify | Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            run_pipeline(frame, classifier, fusion_engine, controller)

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

DEMO_SOURCES = [
    "banana",
    "plastic_bottle",
    "battery",
    "apple",
    "cardboard_box",
    "mobile_phone",
]


def run_demo(classifier, fusion_engine, controller) -> None:
    """Cycle through synthetic demo items."""
    _print("\n[bold]🔄  Running Demo Cycle (6 items) …[/bold]\n")
    for name in DEMO_SOURCES:
        fake_path = f"data/sample_images/{name}.jpg"
        _print(f"\n[dim]→ Item: {name}[/dim]")
        run_pipeline(fake_path, classifier, fusion_engine, controller)
        time.sleep(1.0)

    _print("\n[bold green]✔  Demo complete.[/bold green]")
    _print("\n[bold]Final Sorting Stats:[/bold]")
    for k, v in controller.stats.to_dict().items():
        _print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Smart Waste Segregation and Recycling System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          Run with default sample image
  python main.py --image path/img.jpg     Classify a specific image
  python main.py --webcam                 Use live webcam
  python main.py --dashboard              Launch IoT dashboard on :5000
  python main.py --demo                   Run a 6-item demonstration cycle
  python main.py --tflite                 Use TFLite runtime instead of full TF
  python main.py --serial /dev/ttyUSB0   Send servo commands over serial
        """,
    )
    p.add_argument("--image",     metavar="PATH", help="Image file or URL to classify")
    p.add_argument("--webcam",    action="store_true", help="Use webcam for continuous classification")
    p.add_argument("--dashboard", action="store_true", help="Launch Flask IoT dashboard")
    p.add_argument("--demo",      action="store_true", help="Run demonstration cycle")
    p.add_argument("--tflite",    action="store_true", help="Use TFLite model for inference")
    p.add_argument("--serial",    metavar="PORT",      help="Serial port for Arduino/ESP32 servo control")
    p.add_argument("--port",      type=int, default=5000, help="Dashboard HTTP port (default: 5000)")
    p.add_argument("--no-demo-loop", action="store_true", help="Disable dashboard auto-demo loop")
    p.add_argument("--verbose",   action="store_true", default=True)
    p.add_argument("--quiet",     action="store_true", help="Suppress verbose output")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    verbose = not args.quiet

    _print("""
[bold cyan]╔══════════════════════════════════════════════════╗
║   Smart Waste Segregation & Recycling System     ║
║   AI + Sensor Fusion + IoT Dashboard             ║
╚══════════════════════════════════════════════════╝[/bold cyan]
""" if HAS_RICH else """
╔══════════════════════════════════════════════════╗
║   Smart Waste Segregation & Recycling System     ║
║   AI + Sensor Fusion + IoT Dashboard             ║
╚══════════════════════════════════════════════════╝
""")

    # ── Initialise modules ───────────────────────────────────────────
    WasteClassifier, SensorSimulator, SensorFusionEngine, ServoController, _ = _load_modules()

    logger.info("Initialising classifier …")
    clf = WasteClassifier(use_tflite=args.tflite)

    logger.info("Initialising sensor simulator …")
    sim = SensorSimulator()
    engine = SensorFusionEngine(simulator=sim)

    logger.info("Initialising servo controller …")
    ctrl = ServoController(serial_port=args.serial, simulate=(args.serial is None))

    # ── Mode dispatch ────────────────────────────────────────────────

    if args.dashboard:
        from dashboard import WasteDashboard
        dash = WasteDashboard(
            classifier=clf,
            fusion_engine=engine,
            controller=ctrl,
            port=args.port,
        )
        _print(f"\n[bold green]Dashboard → http://localhost:{args.port}[/bold green]")
        dash.run(demo_loop=not args.no_demo_loop)

    elif args.webcam:
        run_webcam(clf, engine, ctrl)

    elif args.demo:
        run_demo(clf, engine, ctrl)

    else:
        # Single image (default: first sample image or provided path)
        image_source = args.image
        if not image_source:
            # Look for any sample image in data/
            samples = list(Path("data/sample_images").glob("**/*.jpg")) + \
                      list(Path("data/sample_images").glob("**/*.png"))
            if samples:
                image_source = str(samples[0])
                logger.info("No --image specified; using %s", image_source)
            else:
                # Simulate with a synthetic filename
                image_source = "data/sample_images/plastic_bottle.jpg"
                logger.info("No sample images found; using synthetic demo source.")

        run_pipeline(image_source, clf, engine, ctrl, verbose=verbose)

    # Cleanup
    ctrl.cleanup()


if __name__ == "__main__":
    main()
