import os
import uuid
from datetime import timezone

import cv2
import joblib
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from io import BytesIO
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
import spectral
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from matplotlib import pyplot as plt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from skimage.filters import threshold_otsu

from fileProc.models import ProcRecord
from fileUpload.models import HdrFile
from PIL import Image

from .models import PreprocessingMethod, SegmentationMethod, ModelMethod
from .serializers import ProcRecordSerializer

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def clear_temp_images(storage):
    temp_dir = os.path.join(storage.location, 'temp_images')
    if os.path.exists(temp_dir):
        for file_name in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file_name)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}: {e}')


def hdr_to_rgb(request, hdr_id):
    try:
        # 清除临时图像文件夹
        fs = FileSystemStorage()
        clear_temp_images(fs)

        hdr_file = HdrFile.objects.get(pk=hdr_id)
        img = spectral.open_image(hdr_file.file.path)
        rgb_img = spectral.get_rgb(img, bands=(70, 53, 19))  # 假设 RGB 通道为

        # 将图像数据转换为 uint8
        rgb_img = (255 * (rgb_img / rgb_img.max())).astype(np.uint8)

        # 将 Numpy 数组转换为图像
        image_pil = Image.fromarray(rgb_img)
        image_io = BytesIO()
        image_pil.save(image_io, format='JPEG')

        # 生成唯一的文件名
        unique_filename = f"{uuid.uuid4().hex}.jpg"

        # 保存到临时存储
        filename = fs.save(f'temp_images/{unique_filename}', ContentFile(image_io.getvalue()))
        image_url = fs.url(filename)

        # 使用 request.build_absolute_uri() 方法生成完整的 URL
        full_image_url = request.build_absolute_uri(image_url)

        return JsonResponse({'image_url': full_image_url}, status=200)
    except HdrFile.DoesNotExist:
        return JsonResponse({'error': '文件未找到'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def reshape_for_spectral_processing(input_data):
    """重塑三维数据为二维，用于光谱处理"""
    return input_data.reshape(-1, input_data.shape[2])


def revert_shape_after_processing(processed_data, original_shape):
    """将处理后的二维数据重塑回原始的三维形状"""
    return processed_data.reshape(original_shape)


def apply_msc(input_data):
    """多元散射矫正（MSC）"""
    input_data_2d = reshape_for_spectral_processing(input_data)
    # 计算平均光谱
    mean_spectrum = np.mean(input_data_2d, axis=0)
    corrected_spectra = np.zeros_like(input_data_2d)
    for i in range(input_data_2d.shape[0]):
        # 对每个样本执行线性回归
        fit = np.polyfit(mean_spectrum, input_data_2d[i, :], 1)
        corrected_spectra[i, :] = (input_data_2d[i, :] - fit[1]) / fit[0]
    return revert_shape_after_processing(corrected_spectra, input_data.shape)


def apply_snv(input_data):
    """标准正态变量变换（SNV）"""
    input_data_2d = reshape_for_spectral_processing(input_data)
    mean_spectrum = np.mean(input_data_2d, axis=1, keepdims=True)
    std_spectrum = np.std(input_data_2d, axis=1, keepdims=True)

    # 处理标准差为零的情况
    std_spectrum[std_spectrum == 0] = 1e-8

    snv_transformed = (input_data_2d - mean_spectrum) / std_spectrum
    return revert_shape_after_processing(snv_transformed, input_data.shape)


def apply_normalization(input_data):
    """归一化（Normalization）"""
    # 对每个波段进行归一化
    normalized_data = (input_data - np.min(input_data)) / (
            np.max(input_data) - np.min(input_data)
    )
    return normalized_data


def apply_ma_smoothing(input_data, window_length=3):
    """移动平均平滑（MA）"""
    # 确保窗口长度为奇数
    smoothed_data = np.zeros_like(input_data)
    for b in range(input_data.shape[2]):
        smoothed_data[:, :, b] = np.convolve(
            input_data[:, :, b].ravel(),
            np.ones(window_length) / window_length,
            mode="same",
        ).reshape(input_data[:, :, b].shape)
    return smoothed_data


def apply_sg_smoothing(data, window_length=15, polyorder=3):
    """Savitzky-Golay平滑"""
    from scipy.signal import savgol_filter

    # 确保 window_length 是奇数且小于数据的长度
    if window_length >= data.shape[1]:
        window_length = data.shape[1] - 1 if data.shape[1] % 2 == 0 else data.shape[1] - 2

    # 应用滤波器于每个波段
    smoothed_data = savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=1)
    print("sg_smoothing finished")
    return smoothed_data


def apply_Otsu_threshold(input_data, img):
    # 获取波长信息并找到特定波段的索引
    wavelengths = np.array(img.metadata["wavelength"]).astype(float)

    # 找到接近550nm, 660nm, 和800nm的波段索引
    green_band_index = np.abs(np.array(wavelengths) - 550).argmin()
    red_band_index = np.abs(np.array(wavelengths) - 660).argmin()
    nir_band_index = np.abs(np.array(wavelengths) - 800).argmin()
    # 重新组织和转换rgb_simulated为正确的形状
    rgb_simulated = np.stack(
        [
            input_data[:, :, green_band_index],
            input_data[:, :, red_band_index],
            input_data[:, :, nir_band_index],
        ],
        axis=-1,
    )

    # 将数据类型转换为 uint8
    rgb_simulated_uint8 = (
            (rgb_simulated - rgb_simulated.min())
            / (rgb_simulated.max() - rgb_simulated.min())
            * 255
    ).astype("uint8")

    # 选择一个波段进行Otsu阈值分割
    green_band = rgb_simulated_uint8[:, :, 1]
    otsu_thresh = threshold_otsu(green_band)
    binary_mask = green_band > otsu_thresh

    # 初始化GrabCut算法所需的mask
    grabcut_mask = np.where(binary_mask, 3, 2).astype("uint8")  # 前景3，背景2

    # 由于分割效果不理想，手动定义初始掩码区域
    # 例如，如果你知道叶片大致位于图像的中心区域
    center_row, center_col = (
        rgb_simulated_uint8.shape[0] // 2,
        rgb_simulated_uint8.shape[1] // 2,
    )
    size = 60  # 根据叶片的实际大小调整
    grabcut_mask = 2 * np.ones(
        rgb_simulated_uint8.shape[:2], dtype=np.uint8
    )  # 默认为可能的背景
    grabcut_mask[
    center_row - size: center_row + size, center_col - size: center_col + size
    ] = 3  # 叶片区域标记为前景

    # 初始化GrabCut算法其他所需参数
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (
        1,
        1,
        green_band.shape[1] - 1,
        green_band.shape[0] - 1,
    )  # 使用整个图像作为初始矩形

    # 应用GrabCut算法
    cv2.grabCut(
        rgb_simulated_uint8,
        grabcut_mask,
        rect,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_MASK,
    )

    # 转换GrabCut的mask以获取最终分割的前景
    final_mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype("uint8")

    # 对最终掩码应用形态学开运算，去除小的对象
    kernel = np.ones((20, 20), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    segmented_image_al = np.zeros_like(input_data)
    for i in range(input_data.shape[2]):
        segmented_image_al[:, :, i] = input_data[:, :, i] * final_mask
    return segmented_image_al, final_mask


# 加载模型的函数
def load_model(model_type):
    print("model_type: ", model_type)
    model_path = ''
    if model_type == 'pls':
        model_path = os.path.join(settings.MODELS_ROOT, 'pls_model.joblib')
    elif model_type == 'gb':
        model_path = os.path.join(settings.MODELS_ROOT, 'gb_model.joblib')
    elif model_type == 'rf':
        model_path = os.path.join(settings.MODELS_ROOT, 'rf_model.joblib')
    else:
        raise ValueError("Unsupported model type")

    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")


def apply_preprocessing(method, data):
    print("preprocessing_method: ", method)
    if method == 'sg_smoothing':
        return apply_sg_smoothing(data)
    elif method == 'msc':
        return apply_msc(data)
    elif method == 'snv':
        return apply_snv(data)
    elif method == 'normalization':
        return apply_normalization(data)
    elif method == 'ma_smoothing':
        return apply_ma_smoothing(data)
    else:
        raise ValueError("Unsupported preprocessing method")


def apply_segmentation(method, data, img):
    print('segmentation_method: ', method)
    if method == 'otsu':
        return apply_Otsu_threshold(data, img)
    else:
        raise ValueError("Unsupported segmentation method")


def get_methods(request):
    # 预处理方法
    preprocessing_methods = list(PreprocessingMethod.objects.values('label', 'value'))

    # 分割方法
    segmentation_methods = list(SegmentationMethod.objects.values('label', 'value'))

    # 模型选择
    model_methods = list(ModelMethod.objects.values('label', 'value'))

    # 将预处理方法、分割方法和模型方法打包到一个JSON对象中返回
    return JsonResponse({
        'preprocessingMethods': preprocessing_methods,
        'segmentationMethods': segmentation_methods,
        'modelMethods': model_methods
    })


def load_data(image_denoised, selected_indices):
    # 加载数据
    # 从 image_denoised 中选择对应的特征列
    X_selected = image_denoised[:, :, selected_indices]
    return X_selected


import matplotlib

matplotlib.use("SVG")  # 设置Matplotlib的后端为SVG


def model_predict(model, data, mask):
    """模型预测"""
    # 从数据中选择掩码标记为1的部分
    cars_indices_path = os.path.join(settings.DATA_ROOT, "cars_indices.npy")
    selected_indices = np.load(cars_indices_path)
    data_selected = load_data(data, selected_indices)
    print("data_selected.shape: ", data_selected.shape)
    masked_data = np.zeros_like(data_selected)
    for i in range(data_selected.shape[2]):
        masked_data[:, :, i] = data_selected[:, :, i] * mask

    # 将数据重塑为 (n_samples, n_features)，其中 n_samples = 512*512，n_features = selected_indices
    n_samples, height, width = masked_data.shape[0], data_selected.shape[1], data_selected.shape[2]
    data_reshaped = masked_data.reshape(n_samples * height, width)
    print(data_reshaped.shape)

    # 使用模型进行预测
    chlorophyll_content = model.predict(data_reshaped)
    # print(chlorophyll_content)

    # 将预测结果重塑回原来的图像尺寸 (512, 512)
    chlorophyll_content_reshaped = chlorophyll_content.reshape(n_samples, height)

    chlorophyll_content_reshaped = chlorophyll_content_reshaped * mask

    # 使用matplotlib创建图像
    fig, ax = plt.subplots(figsize=(5, 5))  # 设置正方形图像尺寸
    cax = ax.imshow(chlorophyll_content_reshaped, cmap="brg")
    fig.colorbar(cax, label="SPAD值")
    plt.title("叶绿素含量可视化")

    # 将图像保存到 BytesIO 对象
    image_io = BytesIO()
    plt.savefig(image_io, format='svg', bbox_inches='tight')  # 保存为SVG格式
    plt.close(fig)  # 关闭图形，释放资源

    # 使用 FileSystemStorage 保存图像
    fs = FileSystemStorage()
    unique_filename = f"{uuid.uuid4().hex}.svg"  # 修改文件扩展名为svg
    filename = fs.save(f'result_images/{unique_filename}', ContentFile(image_io.getvalue()))
    image_url = fs.url(filename)
    print("image_url: ", image_url)
    return image_url


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def process_image(request):
    model_type = request.data.get('modelSelect')
    print("model_type: ", model_type)
    file_id = request.data.get('fileSelect')
    print("file_id: ", file_id)
    preprocessing_method_value = request.data.get('preMethodSelect')
    print("preprocessing_method_value: ", preprocessing_method_value)
    segmentation_method_value = request.data.get('segMethodSelect')
    print("segmentation_method_value: ", segmentation_method_value)

    try:
        hdr_file = HdrFile.objects.get(pk=file_id)
        model = load_model(model_type)

        img = spectral.open_image(hdr_file.file.path)
        input_data = img.load().astype(float)

        # 先进行分割
        segmented_image, mask = apply_segmentation(segmentation_method_value, input_data, img)
        # 再进行预处理
        preprocessed_data = apply_preprocessing(preprocessing_method_value, segmented_image)

        prediction_image_url = model_predict(model, preprocessed_data, mask)

        full_image_url = request.build_absolute_uri(prediction_image_url)

        # 获取预处理方法、分割方法和模型方法的实例
        preprocessing_method = PreprocessingMethod.objects.get(value=preprocessing_method_value)
        segmentation_method = SegmentationMethod.objects.get(value=segmentation_method_value)
        model_method = ModelMethod.objects.get(value=model_type)

        # 保存处理结果到 ProcRecord 模型中
        proc_record = ProcRecord.objects.create(
            user=request.user,
            hdr_file=hdr_file,
            result=prediction_image_url,
            preprocessing_method=preprocessing_method,
            segmentation_method=segmentation_method,
            model_method=model_method
        )

        return Response({'image_url': full_image_url}, status=200)
    except HdrFile.DoesNotExist:
        return Response({'error': 'File not found'}, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_records(request):
    records = ProcRecord.objects.filter(user=request.user)
    serializer = ProcRecordSerializer(records, many=True, context={'request': request})
    return Response({'records': serializer.data})


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_record(request, id):
    try:
        record = ProcRecord.objects.get(id=id, user=request.user)
        if record.user != request.user:
            return Response({'error': '您没有权限删除!'}, status=403)
        record.delete()
        return Response({'success': '记录删除成功!'}, status=200)
    except ProcRecord.DoesNotExist:
        return Response({'error': '记录没有找到!'}, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=500)