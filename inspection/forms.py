import os

from django import forms
from django.conf import settings

from .models import DEFECT_CLASSES, ReleaseDecision, Review


class MultiFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultiFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultiFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        if isinstance(data, (list, tuple)):
            return [super(MultiFileField, self).clean(item, initial) for item in data]
        return super().clean(data, initial)


class BatchUploadForm(forms.Form):
    name = forms.CharField(max_length=200, required=True)
    notes = forms.CharField(widget=forms.Textarea(attrs={"rows": 2}), required=False)
    files = MultiFileField(required=True)

    def clean_files(self):
        files = self.files.getlist("files") if hasattr(self.files, "getlist") else self.cleaned_data.get("files")
        if not files:
            raise forms.ValidationError("Select at least one image to upload.")
        if len(files) > settings.MAX_BATCH_IMAGES:
            raise forms.ValidationError(
                f"A batch can contain at most {settings.MAX_BATCH_IMAGES} images (got {len(files)})."
            )
        for f in files:
            ext = os.path.splitext(f.name)[1].lower()
            if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
                raise forms.ValidationError(
                    f"'{f.name}' has an unsupported file type. Allowed types: "
                    + ", ".join(settings.ALLOWED_IMAGE_EXTENSIONS)
                )
            content_type = getattr(f, "content_type", None)
            if content_type and content_type not in settings.ALLOWED_IMAGE_CONTENT_TYPES:
                raise forms.ValidationError(f"'{f.name}' does not look like a valid image file.")
            if f.size > settings.MAX_UPLOAD_SIZE_BYTES:
                max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
                raise forms.ValidationError(f"'{f.name}' exceeds the {max_mb:.0f}MB size limit.")
            if f.size == 0:
                raise forms.ValidationError(f"'{f.name}' is empty.")
        return files


class ReviewForm(forms.Form):
    ground_truth_label = forms.ChoiceField(choices=[("", "-- Select label --")] + DEFECT_CLASSES)
    note = forms.CharField(widget=forms.Textarea(attrs={"rows": 3}), required=False)


class ReleaseDecisionForm(forms.ModelForm):
    class Meta:
        model = ReleaseDecision
        fields = ["decision", "rationale"]
        widgets = {
            "rationale": forms.Textarea(attrs={"rows": 4, "placeholder": "Explain the evidence behind this decision..."}),
        }
