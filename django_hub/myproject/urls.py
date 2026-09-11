from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from hub_app import views as hub_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', hub_views.index, name='index'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html', next_page='hub'), name='login'),
    path('register/', hub_views.register, name='register'),
    path('hub/', hub_views.hub, name='hub'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('api/auth-check/', hub_views.auth_check, name='auth_check'),
]