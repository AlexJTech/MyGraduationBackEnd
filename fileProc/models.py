import os

from django.core.files.storage import default_storage
from django.db.models.signals import pre_delete
from django.dispatch import receiver

from fileUpload.models import HdrFile, DatFile
from django.db import models
from django.conf import settings


class PreprocessingMethod(models.Model):
    """预处理方法表"""
    label = models.CharField(max_length=255)
    value = models.CharField(max_length=50)


class SegmentationMethod(models.Model):
    """分割方法表"""
    label = models.CharField(max_length=255)
    value = models.CharField(max_length=50)


class ModelMethod(models.Model):
    """模型方法表"""
    label = models.CharField(max_length=255)
    value = models.CharField(max_length=50)


class ProcRecord(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    hdr_file = models.ForeignKey('fileUpload.HdrFile', on_delete=models.SET_NULL, null=True, blank=True)
    time = models.DateTimeField(auto_now_add=True)
    result = models.TextField()  # To store the URL or result details
    preprocessing_method = models.ForeignKey(PreprocessingMethod, on_delete=models.SET_NULL, null=True, blank=True)
    segmentation_method = models.ForeignKey(SegmentationMethod, on_delete=models.SET_NULL, null=True, blank=True)
    model_method = models.ForeignKey(ModelMethod, on_delete=models.SET_NULL, null=True, blank=True)


@receiver(pre_delete, sender=ProcRecord)
def delete_associated_file(sender, instance, **kwargs):
    # Extract the file path from the URL stored in the result field
    file_url = instance.result

    media_url_prefix = os.path.join(settings.MEDIA_URL, 'result_images')
    if file_url.startswith(media_url_prefix):
        file_path = file_url[len(media_url_prefix):]
        # Print the file path for debugging
        print(f"Deleting file: result_images/{file_path}")
        # Delete the file using default storage
        if default_storage.exists(f'result_images/{file_path}'):
            default_storage.delete(f'result_images/{file_path}')
