"""
classifier.py
=============
Waste Image Classifier using MobileNetV2 (pretrained on ImageNet).

Maps ImageNet categories → Wet / Recyclable / Hazardous waste classes
using a curated label dictionary. Supports:
  - File path input
  - NumPy array (webcam / ESP32-CAM frame)
  - URL input (remote images)

Author : Smart Waste Segregation System
License: MIT
"""

from __future__ import annotations

import logging
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label Mapping  —  ImageNet synset → Waste Category
# ---------------------------------------------------------------------------
# Keys are substrings of ImageNet label names (lower-cased).
# Priority: Hazardous > Recyclable > Wet  (checked in order)

HAZARDOUS_KEYWORDS: list[str] = [
    "battery", "cellular telephone", "mobile phone", "remote control",
    "loudspeaker", "computer keyboard", "laptop", "notebook",
    "hard disc", "modem", "router", "switch", "electric fan",
    "light bulb", "incandescent", "fluorescent", "CRT screen",
    "monitor", "television", "TV", "radiator", "transformer",
    "printed circuit", "power drill", "chain saw", "hand blower",
    "fire engine", "syringe", "pill bottle", "medicine chest",
    "paint can", "bucket", "barrel", "gasoline", "airbrush",
]

RECYCLABLE_KEYWORDS: list[str] = [
    "bottle", "water bottle", "wine bottle", "beer bottle",
    "pop bottle", "plastic bag", "shopping bag",
    "can", "tin can", "beer can", "soda can",
    "cardboard", "carton", "envelope", "paper towel",
    "newspaper", "book", "binder", "menu",
    "cup", "coffee mug", "teapot",
    "container", "canteen", "pitcher", "jar",
    "box", "crate", "hamper",
    "plate", "tray",
    "glass", "goblet",
]

WET_KEYWORDS: list[str] = [
    "banana", "apple", "orange", "lemon", "strawberry",
    "pineapple", "pomegranate", "fig", "jackfruit", "custard apple",
    "mango", "pear", "peach", "nectarine", "grape",
    "mushroom", "broccoli", "cauliflower", "cabbage", "lettuce",
    "spinach", "cucumber", "zucchini", "squash", "artichoke",
    "potato", "sweet potato", "yam", "carrot", "beet",
    "corn", "ear of corn", "pizza", "burrito", "hotdog",
    "hamburger", "meat loaf", "dough", "bread", "bagel",
    "pretzel", "croissant", "waffle", "pancake",
    "soup bowl", "stew", "ice cream", "chocolate",
    "spaghetti squash", "egg",
]

# ---------------------------------------------------------------------------
# ImageNet top-1000 labels (minimal subset for mapping — full list loaded
# dynamically from TF Hub metadata or bundled file)
# ---------------------------------------------------------------------------

_IMAGENET_LABELS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "data/ImageNetLabels.txt"
)
_LABELS_CACHE_PATH = Path(__file__).parent.parent / "models" / "imagenet_labels.txt"

IMG_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_imagenet_labels() -> list[str]:
    """Return the 1001-class ImageNet label list (index 0 = background)."""
    if _LABELS_CACHE_PATH.exists():
        return _LABELS_CACHE_PATH.read_text().strip().splitlines()
    try:
        logger.info("Downloading ImageNet labels …")
        with urllib.request.urlopen(_IMAGENET_LABELS_URL, timeout=10) as resp:
            text = resp.read().decode("utf-8")
        _LABELS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LABELS_CACHE_PATH.write_text(text)
        return text.strip().splitlines()
    except Exception as exc:
        logger.warning("Could not fetch ImageNet labels: %s", exc)
        return []


def _label_to_waste_category(label: str) -> str:
    """Map a single ImageNet label string to a waste category."""
    label_lower = label.lower()
    for kw in HAZARDOUS_KEYWORDS:
        if kw.lower() in label_lower:
            return "Hazardous"
    for kw in RECYCLABLE_KEYWORDS:
        if kw.lower() in label_lower:
            return "Recyclable"
    for kw in WET_KEYWORDS:
        if kw.lower() in label_lower:
            return "Wet"
    return "Recyclable"          # safe default for unknowns


def _preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize + normalise a PIL image for MobileNetV2 input."""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # MobileNetV2 expects pixels in [-1, 1]
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)


# ---------------------------------------------------------------------------
# Main Classifier
# ---------------------------------------------------------------------------

class WasteClassifier:
    """
    MobileNetV2-based waste image classifier.

    Parameters
    ----------
    model_path : str, optional
        Path to a saved TF/TFLite model.  If None the default
        TF Hub MobileNetV2 weights are used.
    use_tflite : bool
        Whether to run TFLite inference instead of full TF.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_tflite: bool = False,
    ) -> None:
        self.use_tflite = use_tflite
        self._model = None
        self._interpreter = None
        self._labels: list[str] = _load_imagenet_labels()
        self._model_path = model_path
        self._loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> "WasteClassifier":
        """Lazy-load the model (call explicitly or on first predict)."""
        if self._loaded:
            return self
        if self.use_tflite:
            self._load_tflite()
        else:
            self._load_tf()
        self._loaded = True
        logger.info("Model loaded successfully.")
        return self

    def _load_tf(self) -> None:
        """Load full TensorFlow / Keras model."""
        try:
            import tensorflow as tf  # noqa: F401  (deferred import)
            from tensorflow.keras.applications import MobileNetV2

            logger.info("Loading MobileNetV2 (ImageNet weights) …")
            self._model = MobileNetV2(weights="imagenet", include_top=True)
            logger.info("MobileNetV2 ready.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load TF model: {exc}") from exc

    def _load_tflite(self) -> None:
        """Load a TFLite model for edge inference."""
        try:
            import tensorflow as tf

            model_path = self._model_path or str(
                Path(__file__).parent.parent / "models" / "mobilenet_v2.tflite"
            )
            if not Path(model_path).exists():
                logger.info("TFLite model not found, falling back to full TF.")
                self.use_tflite = False
                self._load_tf()
                return
            self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._interpreter.allocate_tensors()
            logger.info("TFLite model loaded: %s", model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load TFLite model: {exc}") from exc

    # ------------------------------------------------------------------
    # Image loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_image_from_path(path: Union[str, Path]) -> Image.Image:
        """Load a PIL image from a local file path."""
        return Image.open(path)

    @staticmethod
    def load_image_from_url(url: str) -> Image.Image:
        """Load a PIL image from a remote URL."""
        with urllib.request.urlopen(url, timeout=10) as resp:
            return Image.open(BytesIO(resp.read()))

    @staticmethod
    def load_image_from_array(arr: np.ndarray) -> Image.Image:
        """Convert a BGR/RGB NumPy array (webcam frame) to PIL Image."""
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Assume BGR (OpenCV) → convert to RGB
            arr = arr[:, :, ::-1]
        return Image.fromarray(arr.astype(np.uint8))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_raw(self, preprocessed: np.ndarray) -> np.ndarray:
        """Run the model and return raw probabilities (1001,)."""
        if not self._loaded:
            self.load()

        if self.use_tflite and self._interpreter is not None:
            inp_idx = self._interpreter.get_input_details()[0]["index"]
            out_idx = self._interpreter.get_output_details()[0]["index"]
            self._interpreter.set_tensor(inp_idx, preprocessed)
            self._interpreter.invoke()
            return self._interpreter.get_tensor(out_idx)[0]
        else:
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet_v2 import (
                preprocess_input,
            )
            # Re-normalise for Keras (expects [-1,1] already done above)
            preds = self._model.predict(preprocessed, verbose=0)
            return preds[0]

    def classify(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 5,
    ) -> dict:
        """
        Classify a waste image from any source type.

        Parameters
        ----------
        source : file path | URL string | numpy array | PIL Image
        top_k  : number of top ImageNet predictions to consider

        Returns
        -------
        dict with keys:
            waste_category  – "Wet" | "Recyclable" | "Hazardous"
            confidence      – float in [0, 1]
            imagenet_label  – top ImageNet label string
            top_predictions – list of (label, prob, waste_category)
            image_size      – (W, H) of input
        """
        if not self._loaded:
            self.load()

        # ── Load image ──────────────────────────────────────────────
        if isinstance(source, np.ndarray):
            pil_img = self.load_image_from_array(source)
        elif isinstance(source, Image.Image):
            pil_img = source
        elif isinstance(source, (str, Path)):
            src_str = str(source)
            if src_str.startswith("http://") or src_str.startswith("https://"):
                pil_img = self.load_image_from_url(src_str)
            else:
                pil_img = self.load_image_from_path(src_str)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        original_size = pil_img.size
        preprocessed = _preprocess_image(pil_img)

        # ── Inference ────────────────────────────────────────────────
        probs = self._predict_raw(preprocessed)

        # ── Map to top-k labels ──────────────────────────────────────
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_preds: list[tuple[str, float, str]] = []
        for idx in top_indices:
            if idx < len(self._labels):
                label = self._labels[idx]
            else:
                label = f"class_{idx}"
            prob = float(probs[idx])
            cat = _label_to_waste_category(label)
            top_preds.append((label, prob, cat))

        # ── Aggregate category votes (weighted by probability) ───────
        category_scores: dict[str, float] = {
            "Wet": 0.0,
            "Recyclable": 0.0,
            "Hazardous": 0.0,
        }
        for label, prob, cat in top_preds:
            category_scores[cat] += prob

        best_category = max(category_scores, key=category_scores.__getitem__)  # type: ignore[arg-type]
        total = sum(category_scores.values()) or 1.0
        confidence = category_scores[best_category] / total

        return {
            "waste_category": best_category,
            "confidence": round(confidence, 4),
            "imagenet_label": top_preds[0][0] if top_preds else "unknown",
            "top_predictions": top_preds,
            "image_size": original_size,
            "category_scores": {k: round(v, 4) for k, v in category_scores.items()},
        }

    def classify_with_fallback(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
    ) -> dict:
        """
        Classify with graceful fallback to a simulated result if TF is
        unavailable (useful for testing without GPU/TF installed).
        """
        try:
            return self.classify(source)
        except Exception as exc:
            logger.warning("Classifier fallback triggered: %s", exc)
            return _simulated_classification(source)


# ---------------------------------------------------------------------------
# Simulation fallback (no TF required)
# ---------------------------------------------------------------------------

def _simulated_classification(source) -> dict:
    """
    Return a deterministic simulated classification result.
    Used when TensorFlow is not installed or model fails to load.
    Derives category from filename if possible.
    """
    import random, hashlib

    # Try to derive from filename
    name = ""
    if isinstance(source, (str, Path)):
        name = Path(str(source)).stem.lower()

    for kw in HAZARDOUS_KEYWORDS:
        if kw.lower().replace(" ", "_") in name:
            cat, conf = "Hazardous", 0.82
            break
    else:
        for kw in RECYCLABLE_KEYWORDS:
            if kw.lower().replace(" ", "_") in name:
                cat, conf = "Recyclable", 0.79
                break
        else:
            for kw in WET_KEYWORDS:
                if kw.lower().replace(" ", "_") in name:
                    cat, conf = "Wet", 0.85
                    break
            else:
                # Pseudo-random but reproducible
                seed = int(hashlib.md5(str(source).encode()).hexdigest()[:8], 16)
                random.seed(seed)
                cat = random.choice(["Wet", "Recyclable", "Hazardous"])
                conf = round(random.uniform(0.60, 0.92), 4)

    return {
        "waste_category": cat,
        "confidence": conf,
        "imagenet_label": "[simulated]",
        "top_predictions": [(f"[simulated:{cat}]", conf, cat)],
        "image_size": (224, 224),
        "category_scores": {
            "Wet": conf if cat == "Wet" else round((1 - conf) / 2, 4),
            "Recyclable": conf if cat == "Recyclable" else round((1 - conf) / 2, 4),
            "Hazardous": conf if cat == "Hazardous" else round((1 - conf) / 2, 4),
        },
        "simulated": True,
    }


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    src = sys.argv[1] if len(sys.argv) > 1 else "data/sample_images/plastic_bottle.jpg"
    clf = WasteClassifier()
    result = clf.classify_with_fallback(src)
    print("\n=== Waste Classification Result ===")
    for k, v in result.items():
        if k != "top_predictions":
            print(f"  {k:20s}: {v}")
    print("\n  Top ImageNet Predictions:")
    for label, prob, cat in result["top_predictions"]:
        print(f"    [{cat:12s}]  {prob:.4f}  {label}")
