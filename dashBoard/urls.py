from django.urls import path
from . import views

urlpatterns = [
    path('get_statistics/', views.get_statistics, name='get_statistics'),
]
