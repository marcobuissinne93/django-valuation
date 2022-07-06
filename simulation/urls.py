from django.urls import path
from .views import Main, ValueModel

urlpatterns = [
    path('', Main.as_view(), name='simulate'),
    path('valuation/', ValueModel.as_view()),
]