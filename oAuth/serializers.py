from oAuth.models import CustomUser
from rest_framework import serializers


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['url', 'username', 'email', 'is_staff']
        extra_kwargs = {
            "url": {
                "view_name": "users-detail",
                "lookup_field": "pk"
            }
        }