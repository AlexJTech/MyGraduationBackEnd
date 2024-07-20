import os
from rest_framework import permissions, status, viewsets
from rest_framework.response import Response
from .models import DatFile, HdrFile
from .serializers import DatFileSerializer, HdrFileSerializer

class IsAdminOrIsSelf(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        return request.user.is_staff or obj.user == request.user

class DatFileViewSet(viewsets.ModelViewSet):
    queryset = DatFile.objects.all()
    serializer_class = DatFileSerializer

    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            self.permission_classes = [permissions.IsAuthenticated]
        return super().get_permissions()

    def get_queryset(self):
        return HdrFile.objects.filter(user=self.request.user)

    def create(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({'detail': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        data = {
            'user': request.user.id,  # 添加当前用户ID到数据中
            'file': file_obj
        }
        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def perform_destroy(self, instance):
        file_path = instance.file.path
        if os.path.isfile(file_path):
            os.remove(file_path)
            print("Delete " + file_path + " successfully")
        instance.delete()

class HdrFileViewSet(viewsets.ModelViewSet):
    queryset = HdrFile.objects.all()
    serializer_class = HdrFileSerializer

    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            self.permission_classes = [permissions.IsAuthenticated]
        return super().get_permissions()

    def get_queryset(self):
        return HdrFile.objects.filter(user=self.request.user)

    def create(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({'detail': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        data = {
            'user': request.user.id,  # 添加当前用户ID到数据中
            'file': file_obj
        }
        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def perform_destroy(self, instance):
        file_path = instance.file.path
        if os.path.isfile(file_path):
            os.remove(file_path)
            print("Delete " + file_path + " successfully")
        instance.delete()
