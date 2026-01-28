"""fuel_consumption URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from fuel_app import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='home'),   # root URL → index view
    path('index/', views.index, name='index'),
    path('admin_login/', views.admin_login, name='admin_login'),
    path('admin_home/', views.admin_home, name='admin_home'),
    path('upload_dataset/', views.upload_dataset, name='upload_dataset'),
    
    path('admin_home/preprocess/', views.preprocess_dataset, name='preprocess_dataset'),
    path('admin_home/build/', views.build_model, name='build_model'),
    path('comparison/', views.comparison, name='comparison'),
    path('logout/', views.admin_logout, name='logout'),
    path('user_register/',views.user_register, name='user_register'),
    path('user_login/',views.user_login, name='user_login'),
    path('user_home/', views.user_home, name='user_home'),  # Redirect target
    path('user_logout/', views.user_logout, name='user_logout'),
    path('enter_test_data/', views.enter_test_data, name='enter_test_data'),  # <- Add this



    

]
