from rest_framework import serializers
from .models import DatFile, HdrFile


class DatFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatFile
        fields = '__all__'


class HdrFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = HdrFile
        fields = '__all__'

