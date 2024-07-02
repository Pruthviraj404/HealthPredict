from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from .import views
urlpatterns = [
    path('',views.homepage),
    path('login/', views.login_view, name='logins'),
    path('diabetes/', views.diabities_form, name='diabaties1'),
    path('diabetes-result/', views.diabetes_result, name='diabetes_result'),
    path('heartform/',views.heart_form, name='heartform'),
    path('heartresult/',views.heart_result, name='heartresult1'),
    path('kidneydisease/',views.kidney_form,name="kidneyform"),
    path('kidneyresult/',views.kidney_result,name="kidneyresult1")

]



