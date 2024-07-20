from django.urls import path
from . import views

urlpatterns = [
    path('hdr_to_rgb/<int:hdr_id>/', views.hdr_to_rgb, name='hdr_to_rgb'),
    path('get_methods/', views.get_methods, name='get_methods'),
    path('process_image/', views.process_image, name='process_image'),
    path('get_records/', views.get_records, name='process_video'),
    path('delete_record/<int:id>/', views.delete_record, name='delete_record'),
]
