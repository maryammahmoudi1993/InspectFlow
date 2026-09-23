from django.conf import settings
from django.db import models
from django.urls import reverse


DEFECT_CLASSES = [
    ("ok", "OK / No Defect"),
    ("scratch", "Scratch"),
    ("dent", "Dent"),
    ("discoloration", "Discoloration"),
    ("crack", "Crack"),
]

DEFECT_CLASS_KEYS = [key for key, _ in DEFECT_CLASSES]


class Batch(models.Model):
    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETE = "complete"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_COMPLETE, "Complete"),
    ]

    name = models.CharField(max_length=200)
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name="batches"
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse("batch_detail", args=[self.pk])

    @property
    def image_count(self):
        return self.images.count()


class ProductImage(models.Model):
    batch = models.ForeignKey(Batch, on_delete=models.CASCADE, related_name="images")
    file = models.ImageField(upload_to="batches/%Y/%m/")
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    file_size_bytes = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["batch", "id"]

    def __str__(self):
        return self.original_filename

    @property
    def latest_prediction(self):
        return self.predictions.order_by("-processed_at").first()

    @property
    def review(self):
        return getattr(self, "review_obj", None)

    @property
    def is_reviewed(self):
        review = self.review
        return review is not None and review.status == Review.STATUS_REVIEWED


class ModelVersion(models.Model):
    slug = models.SlugField(max_length=50, unique=True)
    display_name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    is_current = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["slug"]

    def __str__(self):
        return self.display_name


class Prediction(models.Model):
    STATUS_QUEUED = "queued"
    STATUS_PROCESSED = "processed"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_QUEUED, "Queued"),
        (STATUS_PROCESSED, "Processed"),
        (STATUS_FAILED, "Failed"),
    ]

    image = models.ForeignKey(ProductImage, on_delete=models.CASCADE, related_name="predictions")
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name="predictions")
    predicted_class = models.CharField(max_length=30, choices=DEFECT_CLASSES)
    confidence = models.FloatField(help_text="0.0 - 1.0")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_QUEUED)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [("image", "model_version")]
        ordering = ["-processed_at"]

    def __str__(self):
        return f"{self.image} @ {self.model_version} -> {self.predicted_class} ({self.confidence:.2f})"

    @property
    def is_uncertain(self):
        return self.confidence < 0.70


class Review(models.Model):
    STATUS_PENDING = "pending"
    STATUS_REVIEWED = "reviewed"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_REVIEWED, "Reviewed"),
    ]

    image = models.OneToOneField(ProductImage, on_delete=models.CASCADE, related_name="review_obj")
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name="reviews"
    )
    ground_truth_label = models.CharField(max_length=30, choices=DEFECT_CLASSES, blank=True)
    note = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-reviewed_at"]

    def __str__(self):
        return f"Review of {self.image}"


class ReviewAudit(models.Model):
    review = models.ForeignKey(Review, on_delete=models.CASCADE, related_name="audit_entries")
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name="review_audit_entries"
    )
    changed_at = models.DateTimeField(auto_now_add=True)
    field_changed = models.CharField(max_length=50)
    old_value = models.CharField(max_length=200, blank=True)
    new_value = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ["-changed_at"]

    def __str__(self):
        return f"{self.field_changed}: {self.old_value!r} -> {self.new_value!r}"


class EvaluationSet(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    images = models.ManyToManyField(ProductImage, related_name="evaluation_sets", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    @property
    def labeled_images(self):
        """Images in this set that have a reviewed ground-truth label."""
        return self.images.filter(
            review_obj__status=Review.STATUS_REVIEWED
        ).exclude(review_obj__ground_truth_label="")


class ReleaseDecision(models.Model):
    DECISION_APPROVE = "approve"
    DECISION_REJECT = "reject"
    DECISION_NEEDS_REVIEW = "needs_review"
    DECISION_CHOICES = [
        (DECISION_APPROVE, "Approve"),
        (DECISION_REJECT, "Reject"),
        (DECISION_NEEDS_REVIEW, "Needs more review"),
    ]

    candidate_version = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name="candidate_decisions"
    )
    baseline_version = models.ForeignKey(
        ModelVersion, on_delete=models.SET_NULL, null=True, blank=True, related_name="baseline_decisions"
    )
    evaluation_set = models.ForeignKey(EvaluationSet, on_delete=models.CASCADE, related_name="decisions")
    decision = models.CharField(max_length=20, choices=DECISION_CHOICES)
    rationale = models.TextField()
    decided_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name="release_decisions"
    )
    decided_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-decided_at"]

    def __str__(self):
        return f"{self.get_decision_display()} - {self.candidate_version}"
