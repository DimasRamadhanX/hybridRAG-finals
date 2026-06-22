# chatbot/urls.py

from django.urls import path
from . import views

# PASTIKAN: nama variabel adalah urlpatterns dan menggunakan kurung siku []
urlpatterns = [
    path('', views.chat_view, name='chat_endpoint'),
    path('run-analytics/', views.run_analytics_view, name='run_analytics'),
]