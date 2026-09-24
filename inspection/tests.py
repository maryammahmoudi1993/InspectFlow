import io

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
from PIL import Image

from . import demo_classifier
from .evaluation import compare_models, evaluate_model_on_set
from .models import (
    Batch,
    EvaluationSet,
    ModelVersion,
    Prediction,
    ProductImage,
    ReleaseDecision,
    Review,
)


def make_uploaded_image(name="test.jpg"):
    buffer = io.BytesIO()
    Image.new("RGB", (50, 50), (100, 150, 200)).save(buffer, format="JPEG")
    buffer.seek(0)
    return SimpleUploadedFile(name, buffer.read(), content_type="image/jpeg")


class DemoClassifierTests(TestCase):
    def test_classify_is_deterministic(self):
        result_1 = demo_classifier.classify("some-image.jpg", "v1-baseline")
        result_2 = demo_classifier.classify("some-image.jpg", "v1-baseline")
        self.assertEqual(result_1, result_2)

    def test_classify_confidence_in_range(self):
        result = demo_classifier.classify("another-image.jpg", "v2-improved")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


class BatchUploadTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("tester", password="pw12345")
        self.client.force_login(self.user)
        ModelVersion.objects.create(slug="v1-baseline", display_name="V1", is_current=True)
        ModelVersion.objects.create(slug="v2-improved", display_name="V2")

    def test_upload_runs_demo_inference(self):
        response = self.client.post(
            reverse("batch_upload"),
            {"name": "Test batch", "notes": "", "files": [make_uploaded_image()]},
        )
        self.assertEqual(response.status_code, 302)
        batch = Batch.objects.get(name="Test batch")
        self.assertEqual(batch.images.count(), 1)
        image = batch.images.first()
        self.assertEqual(image.predictions.count(), 2)

    def test_upload_rejects_invalid_extension(self):
        bad_file = SimpleUploadedFile("test.txt", b"not an image", content_type="text/plain")
        response = self.client.post(
            reverse("batch_upload"),
            {"name": "Bad batch", "notes": "", "files": [bad_file]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Batch.objects.filter(name="Bad batch").exists())

    def test_upload_error_message_names_the_problem_and_keeps_form(self):
        bad_file = SimpleUploadedFile("report.pdf", b"%PDF-1.4", content_type="application/pdf")
        response = self.client.post(
            reverse("batch_upload"), {"name": "Bad batch", "notes": "", "files": [bad_file]}
        )
        self.assertContains(response, "report.pdf")
        self.assertContains(response, "unsupported file type")
        self.assertContains(response, 'id="upload-btn"')
        self.assertEqual(Batch.objects.count(), 0)

    def test_upload_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse("batch_upload"))
        self.assertEqual(response.status_code, 302)


class ReviewFlowTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("reviewer1", password="pw12345")
        self.client.force_login(self.user)
        self.model_version = ModelVersion.objects.create(slug="v1-baseline", display_name="V1")
        self.batch = Batch.objects.create(name="Batch 1")
        self.image = ProductImage.objects.create(
            batch=self.batch, file=make_uploaded_image(), original_filename="a.jpg"
        )
        Prediction.objects.create(
            image=self.image, model_version=self.model_version, predicted_class="scratch",
            confidence=0.4, status=Prediction.STATUS_PROCESSED,
        )

    def test_review_correction_creates_audit_entry(self):
        url = reverse("review_detail", args=[self.image.pk])
        response = self.client.post(url, {"ground_truth_label": "dent", "note": "corrected"})
        self.assertEqual(response.status_code, 302)
        review = Review.objects.get(image=self.image)
        self.assertEqual(review.ground_truth_label, "dent")
        self.assertEqual(review.status, Review.STATUS_REVIEWED)
        self.assertTrue(review.audit_entries.filter(field_changed="ground_truth_label").exists())


class EvaluationTests(TestCase):
    def setUp(self):
        self.model_a = ModelVersion.objects.create(slug="a", display_name="A")
        self.model_b = ModelVersion.objects.create(slug="b", display_name="B")
        self.batch = Batch.objects.create(name="Eval batch")
        self.images = []
        for i in range(3):
            image = ProductImage.objects.create(
                batch=self.batch, file=make_uploaded_image(f"e{i}.jpg"), original_filename=f"e{i}.jpg"
            )
            Review.objects.create(
                image=image, ground_truth_label="scratch", status=Review.STATUS_REVIEWED
            )
            Prediction.objects.create(
                image=image, model_version=self.model_a, predicted_class="scratch",
                confidence=0.9, status=Prediction.STATUS_PROCESSED,
            )
            Prediction.objects.create(
                image=image, model_version=self.model_b, predicted_class="dent",
                confidence=0.5, status=Prediction.STATUS_PROCESSED,
            )
            self.images.append(image)
        self.eval_set = EvaluationSet.objects.create(name="Set 1")
        self.eval_set.images.set(self.images)

    def test_evaluate_model_on_set_accuracy(self):
        result = evaluate_model_on_set(self.model_a, self.eval_set)
        self.assertEqual(result.evaluated_count, 3)
        self.assertEqual(result.accuracy, 1.0)

        result_b = evaluate_model_on_set(self.model_b, self.eval_set)
        self.assertEqual(result_b.accuracy, 0.0)

    def test_evaluate_excludes_unlabeled_images(self):
        unlabeled = ProductImage.objects.create(
            batch=self.batch, file=make_uploaded_image("u.jpg"), original_filename="u.jpg"
        )
        self.eval_set.images.add(unlabeled)
        result = evaluate_model_on_set(self.model_a, self.eval_set)
        self.assertEqual(result.evaluated_count, 3)

    def test_compare_models_finds_disagreements(self):
        comparison = compare_models(self.model_a, self.model_b, self.eval_set)
        self.assertEqual(len(comparison["disagreements"]), 3)
        self.assertEqual(comparison["total_labeled_images"], 3)

    def test_compare_models_handles_missing_prediction(self):
        extra_image = ProductImage.objects.create(
            batch=self.batch, file=make_uploaded_image("m.jpg"), original_filename="m.jpg"
        )
        Review.objects.create(image=extra_image, ground_truth_label="dent", status=Review.STATUS_REVIEWED)
        Prediction.objects.create(
            image=extra_image, model_version=self.model_a, predicted_class="dent",
            confidence=0.8, status=Prediction.STATUS_PROCESSED,
        )
        self.eval_set.images.add(extra_image)
        comparison = compare_models(self.model_a, self.model_b, self.eval_set)
        self.assertEqual(comparison["both_missing_predictions"], 1)
        self.assertEqual(len(comparison["disagreements"]), 3)


class ReleaseDecisionTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("releaser", password="pw12345")
        self.client.force_login(self.user)
        self.candidate = ModelVersion.objects.create(slug="candidate", display_name="Candidate")
        self.eval_set = EvaluationSet.objects.create(name="Set 1")

    def test_record_decision(self):
        url = reverse("release_candidate_detail", args=[self.candidate.slug])
        response = self.client.post(
            url, {"decision": "needs_review", "rationale": "Not enough coverage yet."}
        )
        self.assertEqual(response.status_code, 302)
        decision = ReleaseDecision.objects.get(candidate_version=self.candidate)
        self.assertEqual(decision.decision, "needs_review")
        self.assertEqual(decision.decided_by, self.user)


class CorrectionFlowsIntoEvaluationTests(TestCase):
    def test_correcting_label_changes_accuracy(self):
        user = get_user_model().objects.create_user("rv", password="pw12345")
        self.client.force_login(user)
        mv = ModelVersion.objects.create(slug="m", display_name="M")
        batch = Batch.objects.create(name="B")
        image = ProductImage.objects.create(batch=batch, file=make_uploaded_image("c.jpg"), original_filename="c.jpg")
        Prediction.objects.create(
            image=image, model_version=mv, predicted_class="scratch",
            confidence=0.9, status=Prediction.STATUS_PROCESSED,
        )
        eval_set = EvaluationSet.objects.create(name="E")
        eval_set.images.add(image)
        url = reverse("review_detail", args=[image.pk])

        self.client.post(url, {"ground_truth_label": "dent", "note": ""})
        self.assertEqual(evaluate_model_on_set(mv, eval_set).accuracy, 0.0)

        self.client.post(url, {"ground_truth_label": "scratch", "note": "fixed"})
        self.assertEqual(evaluate_model_on_set(mv, eval_set).accuracy, 1.0)
        self.assertEqual(Review.objects.get(image=image).audit_entries.filter(field_changed="ground_truth_label").count(), 2)


class ProtectedMediaTests(TestCase):
    def test_media_requires_login(self):
        response = self.client.get("/media/anything.jpg")
        self.assertEqual(response.status_code, 302)

    def test_media_blocks_path_traversal(self):
        user = get_user_model().objects.create_user("mm", password="pw12345")
        self.client.force_login(user)
        response = self.client.get("/media/../manage.py")
        self.assertEqual(response.status_code, 404)


class ClassifierIdentityTests(TestCase):
    def test_stored_path_and_original_filename_classify_identically(self):
        # Uploads are stored under batches/YYYY/MM/, while seeding classifies by
        # the original filename; both must produce the same prediction.
        for slug in ("v1-baseline", "v2-improved"):
            self.assertEqual(
                demo_classifier.classify("demo-image-007.jpg", slug),
                demo_classifier.classify("batches/2026/09/demo-image-007.jpg", slug),
            )


class ReviewQueueFilterTests(TestCase):
    def setUp(self):
        self.client.force_login(get_user_model().objects.create_user("q", password="pw12345"))
        self.v1 = ModelVersion.objects.create(slug="v1", display_name="V1")
        self.v2 = ModelVersion.objects.create(slug="v2", display_name="V2")
        batch = Batch.objects.create(name="B")
        self.image = ProductImage.objects.create(batch=batch, file=make_uploaded_image("q.jpg"), original_filename="q.jpg")
        Prediction.objects.create(image=self.image, model_version=self.v1, predicted_class="scratch",
                                  confidence=0.9, status=Prediction.STATUS_PROCESSED)
        Prediction.objects.create(image=self.image, model_version=self.v2, predicted_class="dent",
                                  confidence=0.6, status=Prediction.STATUS_PROCESSED)

    def test_model_filter_shows_that_models_prediction(self):
        rows = self.client.get(reverse("review_queue"), {"model": "v2"}).context["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1].model_version, self.v2)

    def test_class_filter_matches_displayed_prediction(self):
        url = reverse("review_queue")
        self.assertEqual(len(self.client.get(url, {"model": "v2", "class": "scratch"}).context["rows"]), 0)
        self.assertEqual(len(self.client.get(url, {"model": "v1", "class": "scratch"}).context["rows"]), 1)


import tempfile

from django.core.management import call_command
from django.test import override_settings


class SeedDemoTests(TestCase):
    def test_seed_produces_consistent_shared_evaluation_set(self):
        with tempfile.TemporaryDirectory() as media, override_settings(MEDIA_ROOT=media):
            call_command("seed_demo", "--reset", verbosity=0)
            call_command("seed_demo", "--reset", verbosity=0)  # reset must be repeatable
            self.assertEqual(Batch.objects.count(), 3)
            self.assertEqual(ProductImage.objects.count(), 42)
            eval_set = EvaluationSet.objects.get()
            v1 = evaluate_model_on_set(ModelVersion.objects.get(slug="v1-baseline"), eval_set)
            v2 = evaluate_model_on_set(ModelVersion.objects.get(slug="v2-improved"), eval_set)
            self.assertEqual(v1.evaluated_count, v2.evaluated_count)
            self.assertEqual(v1.evaluated_count, 30)
            self.assertAlmostEqual(v1.accuracy, 0.767, places=2)
            self.assertAlmostEqual(v2.accuracy, 0.933, places=2)
            self.assertEqual(ReleaseDecision.objects.count(), 1)
