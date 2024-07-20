from rest_framework import serializers
from .models import HdrFile, DatFile, ProcRecord


class HdrFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = HdrFile
        fields = '__all__'

class DatFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatFile
        fields = '__all__'

class ProcRecordSerializer(serializers.ModelSerializer):
    hdr_file_name = serializers.CharField(source='hdr_file.file_name', read_only=True)
    preprocessing_method_label = serializers.CharField(source='preprocessing_method.label', read_only=True)
    segmentation_method_label = serializers.CharField(source='segmentation_method.label', read_only=True)
    model_method_label = serializers.CharField(source='model_method.label', read_only=True)
    result = serializers.SerializerMethodField()

    class Meta:
        model = ProcRecord
        fields = ['id', 'time', 'hdr_file_name', 'result', 'preprocessing_method_label', 'segmentation_method_label', 'model_method_label']

    def get_result(self, obj):
        # 如果结果字段存的是经过full_image_url转换的相对路径，可以在这里将其转换为绝对路径
        request = self.context.get('request')
        if obj.result.startswith('/media/'):
            return request.build_absolute_uri(obj.result)
        return obj.result
