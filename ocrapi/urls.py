from django.urls import path
from . import views

urlpatterns = [
    path("image/", views.ImageOCR.as_view(), name="image-ocr"),
    path("pdf/", views.PdfOCR.as_view(), name="pdf-ocr"),
]
