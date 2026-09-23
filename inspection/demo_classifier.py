"""
Deterministic demo inference for InspectFlow.

This module is a stand-in for a real computer-vision classifier. It produces
reproducible, synthetic predictions from an image's bytes so the rest of the
application (review, evaluation, release) can be exercised end to end without
a trained model. It is intentionally the only place that knows how a
"prediction" is produced, so a real classifier can be swapped in later by
replacing `classify()` without touching review/evaluation/release code.

Results are demo-only and must never be presented as real inspection
accuracy.
"""
import hashlib

from .models import DEFECT_CLASS_KEYS

# Each demo model version has its own deterministic "personality": a seed
# salt and a confidence bias, so two versions disagree on some images in a
# reproducible way (useful for the comparison view).
MODEL_PROFILES = {
    "v1-baseline": {"salt": "inspectflow-v1", "confidence_bias": 0.0, "noise": 0.22},
    "v2-improved": {"salt": "inspectflow-v2", "confidence_bias": 0.08, "noise": 0.14},
}

DEFAULT_PROFILE = {"salt": "inspectflow-default", "confidence_bias": 0.0, "noise": 0.2}


def _hash_to_unit_float(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def true_label_for_image(image_identity: str) -> str:
    """Deterministic synthetic 'ground truth' used only to generate demo
    content (seed data). A real system would never derive ground truth this
    way; genuine ground truth comes from human review.
    """
    value = _hash_to_unit_float("true-label", image_identity)
    index = int(value * len(DEFECT_CLASS_KEYS)) % len(DEFECT_CLASS_KEYS)
    return DEFECT_CLASS_KEYS[index]


def classify(image_identity: str, model_slug: str) -> dict:
    """Return a deterministic synthetic prediction for the given image.

    `image_identity` should be a stable string identifying the image
    (e.g. its stored filename). The same identity + model_slug always
    yields the same result.
    """
    profile = MODEL_PROFILES.get(model_slug, DEFAULT_PROFILE)
    truth = true_label_for_image(image_identity)

    correctness_roll = _hash_to_unit_float(profile["salt"], image_identity, "correct")
    # Better-behaved profiles (lower noise) are more likely to predict the
    # synthetic truth; this is what makes the two demo versions comparably
    # different in the model-comparison view.
    if correctness_roll > profile["noise"]:
        predicted_class = truth
    else:
        offset_roll = _hash_to_unit_float(profile["salt"], image_identity, "offset")
        others = [c for c in DEFECT_CLASS_KEYS if c != truth]
        predicted_class = others[int(offset_roll * len(others)) % len(others)]

    confidence_roll = _hash_to_unit_float(profile["salt"], image_identity, "confidence")
    base_confidence = 0.55 + confidence_roll * 0.4
    if predicted_class != truth:
        base_confidence *= 0.75
    confidence = max(0.05, min(0.99, base_confidence + profile["confidence_bias"]))

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
    }
