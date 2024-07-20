from django.core.management.base import BaseCommand
from fileProc.models import PreprocessingMethod, SegmentationMethod, ModelMethod

class Command(BaseCommand):
    help = 'Import methods into the database'

    def handle(self, *args, **options):
        preprocessing_methods = [
            {'label': '多元散射矫正（MSC）', 'value': 'msc'},
            {'label': '标准正态变量变换（SNV）', 'value': 'snv'},
            {'label': '归一化', 'value': 'normalization'},
            {'label': '移动平均平滑（MA）', 'value': 'ma_smoothing'},
            {'label': 'Savitzky-Golay平滑', 'value': 'sg_smoothing'}
        ]

        segmentation_methods = [
            {'label': 'Otsu 阈值分割 + GrabCut 算法', 'value': 'otsu'}
        ]

        model_methods = [
            {'label': 'PLS 模型', 'value': 'pls'},
            {'label': 'RF 随机森林模型', 'value': 'rf'},
            {'label': 'GB 梯度生成树模型', 'value': 'gb'}
        ]

        # Clear existing data
        PreprocessingMethod.objects.all().delete()
        SegmentationMethod.objects.all().delete()
        ModelMethod.objects.all().delete()

        # Import new data
        for method in preprocessing_methods:
            PreprocessingMethod.objects.create(**method)

        for method in segmentation_methods:
            SegmentationMethod.objects.create(**method)

        for method in model_methods:
            ModelMethod.objects.create(**method)

        self.stdout.write(self.style.SUCCESS('Successfully imported methods'))
