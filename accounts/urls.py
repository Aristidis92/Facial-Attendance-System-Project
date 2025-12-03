# accounts/urls.py ✅ (correct version)
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
import csv
urlpatterns = [
    path('register/', views.student_register, name='student_register'),
    path('register/teacher/', views.teacher_register, name='teacher_register'),
    path('register/admin/', views.admin_register, name='admin_register'),
    path('profile/', views.student_profile, name='student_profile'),
    path('redirect/', views.role_redirect, name='role_redirect'),
    path('teacher/dashboard/', views.teacher_dashboard, name='teacher_dashboard'),
    path('dashboard/export/', views.export_attendance_csv, name='export_attendance_csv'),
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin/users/', views.manage_users, name='manage_users'),
    path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='accounts/logout.html'), name='logout'),
    path('admin/users/', views.user_list, name='user_list'),
    path('user/<int:user_id>/edit/', views.edit_user, name='edit_user'),
    path('user/<int:user_id>/delete/', views.delete_user, name='delete_user'),
    path('student/update-face/', views.update_face_data, name='update_face'),


]
