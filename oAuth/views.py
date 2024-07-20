from django.shortcuts import render
from rest_framework import viewsets, status
from oAuth.models import CustomUser
from rest_framework.response import Response
from oAuth.serializers import UserSerializer


# Create your views here.

class UserInfoViewSet(viewsets.ViewSet):
    queryset = CustomUser.objects.all().order_by('id')
    http_method_names = ['get']

    def list(self, request, *args, **kwargs):
        # print('ok')
        user_info = CustomUser.objects.filter(id=request.user.id).values()[0]
        role = request.user.roles
        if role == 0:
            user_info['roles'] = ['admin']
        else:
            user_info['roles'] = ['user']
        return Response(user_info)


class UserViewSet(viewsets.ModelViewSet):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer

    # 实现封装方法的重写

    # list: get
    # create: post
    def list(self, request, *args, **kwargs):
        # 设置排除admin用户

        user = request.user
        if user.roles == 1:
            self.queryset = self.queryset.exclude(roles=0)
        queryset = self.filter_queryset(self.get_queryset())

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

class UserCreateViewSet(viewsets.ModelViewSet):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer

    # POST 方法
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def perform_create(self, serializer):
        serializer.save()

    def get_success_headers(self, data):
        try:
            return {'Location': str(data[api_settings.URL_FIELD_NAME])}
        except (TypeError, KeyError):
            return {}

