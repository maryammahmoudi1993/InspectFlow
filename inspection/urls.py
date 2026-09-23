from django.contrib.auth.views import LogoutView
from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("login/", views.InspectFlowLoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("batches/upload/", views.batch_upload, name="batch_upload"),
    path("batches/<int:pk>/", views.batch_detail, name="batch_detail"),
    path("review/", views.review_queue, name="review_queue"),
    path("review/<int:pk>/", views.review_detail, name="review_detail"),
    path("compare/", views.model_comparison, name="model_comparison"),
    path("release/", views.release_candidates, name="release_candidates"),
    path("release/<slug:slug>/", views.release_candidate_detail, name="release_candidate_detail"),
]
