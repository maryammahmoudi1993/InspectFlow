from django.contrib import admin

from .models import (
    Batch,
    EvaluationSet,
    ModelVersion,
    Prediction,
    ProductImage,
    ReleaseDecision,
    Review,
    ReviewAudit,
)

admin.site.register(Batch)
admin.site.register(ProductImage)
admin.site.register(ModelVersion)
admin.site.register(Prediction)
admin.site.register(Review)
admin.site.register(ReviewAudit)
admin.site.register(EvaluationSet)
admin.site.register(ReleaseDecision)
