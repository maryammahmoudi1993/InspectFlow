from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.db import transaction
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from . import demo_classifier
from .evaluation import compare_models, evaluate_model_on_set
from .forms import BatchUploadForm, ReleaseDecisionForm, ReviewForm
from .models import (
    Batch,
    DEFECT_CLASSES,
    EvaluationSet,
    ModelVersion,
    Prediction,
    ProductImage,
    ReleaseDecision,
    Review,
    ReviewAudit,
)


class InspectFlowLoginView(LoginView):
    template_name = "inspection/login.html"


@login_required
def dashboard(request):
    batches = Batch.objects.all()[:6]
    pending_review_count = ProductImage.objects.exclude(
        review_obj__status=Review.STATUS_REVIEWED
    ).count()
    model_versions = ModelVersion.objects.all()
    evaluation_sets = EvaluationSet.objects.all()
    recent_decisions = ReleaseDecision.objects.select_related("candidate_version")[:5]
    context = {
        "batches": batches,
        "pending_review_count": pending_review_count,
        "model_versions": model_versions,
        "evaluation_sets": evaluation_sets,
        "recent_decisions": recent_decisions,
        "batch_count": Batch.objects.count(),
        "image_count": ProductImage.objects.count(),
    }
    return render(request, "inspection/dashboard.html", context)


def _run_demo_inference(image: ProductImage):
    """Run every registered demo model version against a single image and
    store the resulting predictions. Kept separate so batch upload logic
    doesn't need to know how inference works.
    """
    identity = image.file.name
    for model_version in ModelVersion.objects.all():
        result = demo_classifier.classify(identity, model_version.slug)
        Prediction.objects.update_or_create(
            image=image,
            model_version=model_version,
            defaults={
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "status": Prediction.STATUS_PROCESSED,
                "processed_at": timezone.now(),
            },
        )


@login_required
def batch_upload(request):
    if request.method == "POST":
        form = BatchUploadForm(request.POST, request.FILES)
        if form.is_valid():
            with transaction.atomic():
                batch = Batch.objects.create(
                    name=form.cleaned_data["name"],
                    notes=form.cleaned_data["notes"],
                    uploaded_by=request.user,
                    status=Batch.STATUS_PROCESSING,
                )
                files = form.cleaned_data["files"]
                created_images = []
                for f in files:
                    image = ProductImage.objects.create(
                        batch=batch,
                        file=f,
                        original_filename=f.name,
                        file_size_bytes=f.size,
                    )
                    created_images.append(image)
                for image in created_images:
                    _run_demo_inference(image)
                batch.status = Batch.STATUS_COMPLETE
                batch.save(update_fields=["status"])
            messages.success(
                request,
                f"Batch '{batch.name}' uploaded: {len(created_images)} image(s) processed by the demo classifier.",
            )
            return redirect("batch_detail", pk=batch.pk)
    else:
        form = BatchUploadForm()
    return render(request, "inspection/batch_upload.html", {"form": form})


@login_required
def batch_detail(request, pk):
    batch = get_object_or_404(Batch, pk=pk)
    images = batch.images.select_related("review_obj").prefetch_related("predictions__model_version")
    return render(request, "inspection/batch_detail.html", {"batch": batch, "images": images})


@login_required
def review_queue(request):
    images = ProductImage.objects.select_related("batch", "review_obj").prefetch_related(
        "predictions__model_version"
    )

    batch_id = request.GET.get("batch")
    predicted_class = request.GET.get("class")
    model_slug = request.GET.get("model")
    status = request.GET.get("status")
    sort = request.GET.get("sort", "uncertain")

    if batch_id:
        images = images.filter(batch_id=batch_id)
    if predicted_class:
        images = images.filter(predictions__predicted_class=predicted_class)
    if model_slug:
        images = images.filter(predictions__model_version__slug=model_slug)
    if status == "pending":
        images = images.exclude(review_obj__status=Review.STATUS_REVIEWED)
    elif status == "reviewed":
        images = images.filter(review_obj__status=Review.STATUS_REVIEWED)

    images = images.distinct()

    rows = []
    for image in images:
        prediction = image.latest_prediction
        rows.append((image, prediction))

    if sort == "uncertain":
        rows.sort(key=lambda pair: (pair[1].confidence if pair[1] else 1.0))
    elif sort == "confidence_desc":
        rows.sort(key=lambda pair: (pair[1].confidence if pair[1] else 0.0), reverse=True)
    elif sort == "recent":
        rows.sort(key=lambda pair: pair[0].uploaded_at, reverse=True)

    context = {
        "rows": rows,
        "batches": Batch.objects.all(),
        "model_versions": ModelVersion.objects.all(),
        "defect_classes": DEFECT_CLASSES,
        "selected_batch": batch_id or "",
        "selected_class": predicted_class or "",
        "selected_model": model_slug or "",
        "selected_status": status or "",
        "selected_sort": sort,
    }
    return render(request, "inspection/review_queue.html", context)


@login_required
def review_detail(request, pk):
    image = get_object_or_404(
        ProductImage.objects.select_related("batch", "review_obj").prefetch_related(
            "predictions__model_version"
        ),
        pk=pk,
    )
    review, _ = Review.objects.get_or_create(image=image)

    if request.method == "POST":
        form = ReviewForm(request.POST)
        if form.is_valid():
            new_label = form.cleaned_data["ground_truth_label"]
            new_note = form.cleaned_data["note"]
            old_label = review.ground_truth_label
            old_note = review.note

            with transaction.atomic():
                review.ground_truth_label = new_label
                review.note = new_note
                review.status = Review.STATUS_REVIEWED
                review.reviewer = request.user
                review.reviewed_at = timezone.now()
                review.save()

                if old_label != new_label:
                    ReviewAudit.objects.create(
                        review=review,
                        changed_by=request.user,
                        field_changed="ground_truth_label",
                        old_value=old_label,
                        new_value=new_label,
                    )
                if old_note != new_note:
                    ReviewAudit.objects.create(
                        review=review,
                        changed_by=request.user,
                        field_changed="note",
                        old_value=old_note,
                        new_value=new_note,
                    )
            messages.success(request, "Review saved.")
            next_url = request.POST.get("next")
            return redirect(next_url or "review_queue")
    else:
        form = ReviewForm(initial={"ground_truth_label": review.ground_truth_label, "note": review.note})

    predictions = image.predictions.select_related("model_version").all()
    audit_entries = review.audit_entries.select_related("changed_by").all()
    context = {
        "image": image,
        "review": review,
        "form": form,
        "predictions": predictions,
        "audit_entries": audit_entries,
        "next": request.GET.get("next", ""),
    }
    return render(request, "inspection/review_detail.html", context)


@login_required
def model_comparison(request):
    model_versions = ModelVersion.objects.all()
    evaluation_sets = EvaluationSet.objects.all()

    version_a_slug = request.GET.get("model_a")
    version_b_slug = request.GET.get("model_b")
    eval_set_id = request.GET.get("evaluation_set")

    comparison = None
    if version_a_slug and version_b_slug and eval_set_id:
        model_a = get_object_or_404(ModelVersion, slug=version_a_slug)
        model_b = get_object_or_404(ModelVersion, slug=version_b_slug)
        evaluation_set = get_object_or_404(EvaluationSet, pk=eval_set_id)
        comparison = compare_models(model_a, model_b, evaluation_set)

    context = {
        "model_versions": model_versions,
        "evaluation_sets": evaluation_sets,
        "comparison": comparison,
        "selected_a": version_a_slug or "",
        "selected_b": version_b_slug or "",
        "selected_eval_set": eval_set_id or "",
        "defect_classes": DEFECT_CLASSES,
    }
    return render(request, "inspection/model_comparison.html", context)


@login_required
def release_candidates(request):
    model_versions = ModelVersion.objects.all()
    decisions = ReleaseDecision.objects.select_related("candidate_version", "decided_by")
    return render(
        request,
        "inspection/release_list.html",
        {"model_versions": model_versions, "decisions": decisions},
    )


@login_required
def release_candidate_detail(request, slug):
    candidate = get_object_or_404(ModelVersion, slug=slug)
    baseline = ModelVersion.objects.filter(is_current=True).exclude(pk=candidate.pk).first()
    evaluation_set = EvaluationSet.objects.first()

    candidate_result = None
    baseline_result = None
    total_images = ProductImage.objects.count()
    reviewed_count = Review.objects.filter(status=Review.STATUS_REVIEWED).count()

    if evaluation_set:
        candidate_result = evaluate_model_on_set(candidate, evaluation_set)
        if baseline:
            baseline_result = evaluate_model_on_set(baseline, evaluation_set)

    if request.method == "POST":
        form = ReleaseDecisionForm(request.POST)
        if form.is_valid():
            decision = form.save(commit=False)
            decision.candidate_version = candidate
            decision.baseline_version = baseline
            decision.evaluation_set = evaluation_set
            decision.decided_by = request.user
            decision.save()
            messages.success(request, "Release decision recorded.")
            return redirect("release_candidate_detail", slug=slug)
    else:
        form = ReleaseDecisionForm()

    past_decisions = candidate.candidate_decisions.select_related("decided_by")

    context = {
        "candidate": candidate,
        "baseline": baseline,
        "evaluation_set": evaluation_set,
        "candidate_result": candidate_result,
        "baseline_result": baseline_result,
        "total_images": total_images,
        "reviewed_count": reviewed_count,
        "form": form,
        "past_decisions": past_decisions,
    }
    return render(request, "inspection/release_detail.html", context)
