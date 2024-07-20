from django.db import models
from django.conf import settings
from django.utils import timezone


class DatFile(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    file = models.FileField(upload_to='spectrals/', verbose_name="DAT File")
    file_name = models.CharField(max_length=255, verbose_name="Original File Name", blank=True)
    uploaded_at = models.DateTimeField(default=timezone.now, verbose_name="Uploaded At")

    def save(self, *args, **kwargs):
        # 直接使用上传的文件名更新 file_name 字段
        self.file_name = self.file.name
        super().save(*args, **kwargs)

class HdrFile(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    file = models.FileField(upload_to='spectrals/', verbose_name="HDR File")
    file_name = models.CharField(max_length=255, verbose_name="Original File Name", blank=True)
    uploaded_at = models.DateTimeField(default=timezone.now, verbose_name="Uploaded At")

    def save(self, *args, **kwargs):
        # 直接使用上传的文件名更新 file_name 字段
        self.file_name = self.file.name
        super().save(*args, **kwargs)

