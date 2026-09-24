"""
Evaluation and comparison logic. Operates only on reviewed (labeled) images
and stored predictions -- it never calls the demo classifier directly, so it
would work unchanged against predictions from a real model.
"""
from dataclasses import dataclass, field

from .models import DEFECT_CLASS_KEYS, Prediction


@dataclass
class ClassMetrics:
    label: str
    support: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self):
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else None

    @property
    def recall(self):
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else None


@dataclass
class EvaluationResult:
    model_version = None
    evaluated_count: int = 0
    skipped_missing_prediction: int = 0
    accuracy: float = 0.0
    class_metrics: dict = field(default_factory=dict)
    confusion_matrix: dict = field(default_factory=dict)  # {true_label: {pred_label: count}}
    unlabeled_count: int = 0
    total_in_set: int = 0

    @property
    def matrix_rows(self):
        return [
            (true, [self.confusion_matrix[true][pred] for pred in DEFECT_CLASS_KEYS])
            for true in DEFECT_CLASS_KEYS
        ]

    @property
    def class_keys(self):
        return DEFECT_CLASS_KEYS


def evaluate_model_on_set(model_version, evaluation_set):
    """Compute metrics for one model version against the evaluation set's
    labeled images. Images without a reviewed ground-truth label, or without
    a prediction from this model version, are excluded and counted
    explicitly rather than silently dropped.
    """
    labeled_images = list(evaluation_set.labeled_images)
    predictions = {
        p.image_id: p
        for p in Prediction.objects.filter(
            image__in=labeled_images, model_version=model_version, status=Prediction.STATUS_PROCESSED
        )
    }

    result = EvaluationResult()
    result.model_version = model_version
    class_metrics = {key: ClassMetrics(label=key) for key in DEFECT_CLASS_KEYS}
    confusion = {t: {p: 0 for p in DEFECT_CLASS_KEYS} for t in DEFECT_CLASS_KEYS}

    correct = 0
    evaluated = 0
    for image in labeled_images:
        truth = image.review_obj.ground_truth_label
        prediction = predictions.get(image.id)
        if prediction is None:
            result.skipped_missing_prediction += 1
            continue
        evaluated += 1
        predicted = prediction.predicted_class
        class_metrics[truth].support += 1
        confusion[truth][predicted] += 1
        if predicted == truth:
            correct += 1
            class_metrics[truth].true_positives += 1
        else:
            class_metrics[truth].false_negatives += 1
            class_metrics[predicted].false_positives += 1

    result.total_in_set = evaluation_set.images.count()
    result.unlabeled_count = result.total_in_set - len(labeled_images)
    result.evaluated_count = evaluated
    result.accuracy = (correct / evaluated) if evaluated else None
    result.class_metrics = class_metrics
    result.confusion_matrix = confusion
    return result


@dataclass
class DisagreementCase:
    image = None
    label_a = None
    confidence_a = None
    label_b = None
    confidence_b = None
    ground_truth = None


def compare_models(model_a, model_b, evaluation_set):
    """Return per-model evaluation results plus the list of labeled images
    where the two models disagree, using only the shared, fully-labeled
    subset so the comparison is apples-to-apples.
    """
    labeled_images = list(evaluation_set.labeled_images)
    preds_a = {
        p.image_id: p
        for p in Prediction.objects.filter(
            image__in=labeled_images, model_version=model_a, status=Prediction.STATUS_PROCESSED
        )
    }
    preds_b = {
        p.image_id: p
        for p in Prediction.objects.filter(
            image__in=labeled_images, model_version=model_b, status=Prediction.STATUS_PROCESSED
        )
    }

    result_a = evaluate_model_on_set(model_a, evaluation_set)
    result_b = evaluate_model_on_set(model_b, evaluation_set)

    disagreements = []
    both_missing = 0
    for image in labeled_images:
        pa = preds_a.get(image.id)
        pb = preds_b.get(image.id)
        if pa is None or pb is None:
            both_missing += 1
            continue
        if pa.predicted_class != pb.predicted_class:
            case = DisagreementCase()
            case.image = image
            case.label_a = pa.predicted_class
            case.confidence_a = pa.confidence
            case.label_b = pb.predicted_class
            case.confidence_b = pb.confidence
            case.ground_truth = image.review_obj.ground_truth_label
            disagreements.append(case)

    return {
        "evaluation_set": evaluation_set,
        "total_labeled_images": len(labeled_images),
        "both_missing_predictions": both_missing,
        "result_a": result_a,
        "result_b": result_b,
        "results_list": [result_a, result_b],
        "disagreements": disagreements,
    }
