from rest_framework import serializers


class ChatCompletionSerializer(serializers.Serializer):
    name = serializers.CharField()
    dob = serializers.CharField()
    phone = serializers.CharField()
    adhaar = serializers.CharField()
    address = serializers.CharField(allow_null=True)
    type = serializers.CharField()
    language = serializers.CharField()
    gender = serializers.CharField()
