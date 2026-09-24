"""
Seed InspectFlow with a deterministic, synthetic demo dataset: two model
versions, three image batches, demo predictions, a portion of pre-reviewed
images, an evaluation set, and one illustrative release decision.

All images are generated in-process with Pillow -- simple colored panels
with a drawn "defect" marker -- so the demo has no external data
dependencies or licensing concerns.
"""
import hashlib
import io
import random
import shutil
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from PIL import Image, ImageDraw

from inspection import demo_classifier
from inspection.models import (
    Batch,
    DEFECT_CLASS_KEYS,
    EvaluationSet,
    ModelVersion,
    Prediction,
    ProductImage,
    ReleaseDecision,
    Review,
)

PANEL_COLORS = {
    "ok": (198, 224, 201),
    "scratch": (233, 214, 107),
    "dent": (222, 184, 135),
    "discoloration": (216, 148, 148),
    "crack": (176, 176, 200),
}

BATCH_DEFINITIONS = [
    ("Line A - Morning Run", 16),
    ("Line B - Afternoon Run", 14),
    ("Line A - QA Spot Check", 12),
]


def render_synthetic_image(seed_text: str, label: str) -> ContentFile:
    """Draw a small deterministic synthetic 'product photo': a colored
    panel with a shape representing the (seed-derived) defect class.
    """
    rng = random.Random(hashlib.sha256(seed_text.encode()).hexdigest())
    size = (320, 240)
    bg = PANEL_COLORS.get(label, (200, 200, 200))
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)

    # base panel texture
    for _ in range(6):
        x0 = rng.randint(0, size[0])
        y0 = rng.randint(0, size[1])
        x1 = min(size[0], x0 + rng.randint(10, 40))
        y1 = min(size[1], y0 + rng.randint(10, 40))
        shade = tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in bg)
        draw.rectangle([x0, y0, x1, y1], fill=shade)

    cx, cy = size[0] // 2, size[1] // 2
    if label == "scratch":
        draw.line([(cx - 80, cy - 40), (cx + 80, cy + 40)], fill=(90, 90, 90), width=4)
    elif label == "dent":
        draw.ellipse([cx - 30, cy - 20, cx + 30, cy + 20], outline=(90, 60, 30), width=4)
    elif label == "discoloration":
        draw.ellipse([cx - 50, cy - 35, cx + 50, cy + 35], fill=(150, 60, 60))
    elif label == "crack":
        draw.line([(cx - 60, cy - 50), (cx - 10, cy), (cx - 55, cy + 55)], fill=(40, 40, 40), width=3)

    draw.text((8, 8), f"InspectFlow demo · {label}", fill=(60, 60, 60))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return ContentFile(buffer.getvalue())


class Command(BaseCommand):
    help = "Seed a deterministic synthetic demo dataset for InspectFlow."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset", action="store_true", help="Delete existing demo data before reseeding."
        )

    def handle(self, *args, **options):
        User = get_user_model()

        if options["reset"]:
            self.stdout.write("Clearing existing data...")
            ReleaseDecision.objects.all().delete()
            EvaluationSet.objects.all().delete()
            Prediction.objects.all().delete()
            Review.objects.all().delete()
            ProductImage.objects.all().delete()
            Batch.objects.all().delete()
            ModelVersion.objects.all().delete()
            shutil.rmtree(Path(settings.MEDIA_ROOT) / 'batches', ignore_errors=True)

        with transaction.atomic():
            reviewer, created = User.objects.get_or_create(
                username="reviewer",
                defaults={"is_staff": True, "email": "reviewer@example.com"},
            )
            if created:
                reviewer.set_password("inspectflow-demo")
                reviewer.save()
                self.stdout.write("Created demo user 'reviewer' / 'inspectflow-demo'.")

            v1, _ = ModelVersion.objects.update_or_create(
                slug="v1-baseline",
                defaults={
                    "display_name": "Demo Classifier v1 (baseline)",
                    "description": "Deterministic synthetic baseline classifier used for demo purposes only.",
                    "is_current": True,
                },
            )
            v2, _ = ModelVersion.objects.update_or_create(
                slug="v2-improved",
                defaults={
                    "display_name": "Demo Classifier v2 (candidate)",
                    "description": "Deterministic synthetic candidate classifier with lower simulated noise. Demo only.",
                    "is_current": False,
                },
            )

            all_images = []
            counter = 0
            for batch_name, count in BATCH_DEFINITIONS:
                batch = Batch.objects.create(
                    name=batch_name,
                    uploaded_by=reviewer,
                    status=Batch.STATUS_COMPLETE,
                    notes="Synthetic demo batch generated by seed_demo.",
                )
                for i in range(count):
                    counter += 1
                    identity = f"demo-image-{counter:03d}.jpg"
                    true_label = demo_classifier.true_label_for_image(identity)
                    content = render_synthetic_image(identity, true_label)
                    image = ProductImage(
                        batch=batch,
                        original_filename=identity,
                        width=320,
                        height=240,
                        file_size_bytes=content.size,
                    )
                    image.file.save(identity, content, save=True)
                    all_images.append((image, true_label))
                self.stdout.write(f"Created batch '{batch_name}' with {count} images.")

            for image, _true_label in all_images:
                for model_version in (v1, v2):
                    result = demo_classifier.classify(image.original_filename, model_version.slug)
                    Prediction.objects.create(
                        image=image,
                        model_version=model_version,
                        predicted_class=result["predicted_class"],
                        confidence=result["confidence"],
                        status=Prediction.STATUS_PROCESSED,
                        processed_at=timezone.now(),
                    )

            # Pre-review roughly 70% of images to make the evaluation set
            # and review queue immediately useful; leave the rest pending
            # so the review workflow has real work to do.
            reviewed_images = []
            for index, (image, true_label) in enumerate(all_images):
                if index % 10 < 7:
                    review = Review.objects.create(
                        image=image,
                        reviewer=reviewer,
                        ground_truth_label=true_label,
                        status=Review.STATUS_REVIEWED,
                        reviewed_at=timezone.now(),
                        note="Confirmed during initial demo QA pass." if index % 3 == 0 else "",
                    )
                    reviewed_images.append(image)

            evaluation_set = EvaluationSet.objects.create(
                name="Initial QA Evaluation Set",
                description="All demo images reviewed during the initial synthetic QA pass.",
            )
            evaluation_set.images.set(reviewed_images)

            ReleaseDecision.objects.create(
                candidate_version=v1,
                baseline_version=None,
                evaluation_set=evaluation_set,
                decision=ReleaseDecision.DECISION_APPROVE,
                rationale=(
                    "Baseline demo classifier approved as the initial current version for portfolio "
                    "demonstration purposes only; not evaluated against any real manufacturing data."
                ),
                decided_by=reviewer,
            )

        self.stdout.write(self.style.SUCCESS(
            f"Seed complete: {len(all_images)} images, {len(reviewed_images)} pre-reviewed, "
            f"evaluation set '{evaluation_set.name}'."
        ))
