from django.contrib import admin
from django.urls import path, include
from . import views



urlpatterns = [
    path('analizar-universidad/<str:universidad_id>/', views.analizar_universidad, name='analizar_universidad'),
    path('obtener_palabras_de_interes/<str:universidad_id>/', views.obtener_palabras_de_interes, name='obtener_palabras_de_interes'),
    path('detener_procesamiento/', views.stop_processing_view, name='detener_procesamiento'),
    
]