from django.urls import path, include
from rest_framework.routers import DefaultRouter

from fileUpload.views import DatFileViewSet, HdrFileViewSet

router = DefaultRouter()
router.register(r'dat', DatFileViewSet)
router.register(r'hdr', HdrFileViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
