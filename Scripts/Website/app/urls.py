from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('prediccion', views.prediction),
    path('prediccion/clasificacion', views.classification),
    path('prediccion/segmentacion', views.segmentation),
    path('documentacion', views.documentation),
    path('manual', views.manual),
    path('instalacion', views.instalacion),
]
