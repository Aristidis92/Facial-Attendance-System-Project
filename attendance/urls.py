from django.urls import path
from . import views
from .views import update_session_records

urlpatterns = [
    path('teacher/dashboard/', views.teacher_dashboard, name='teacher_dashboard'),
    path('mark/<int:unit_id>/', views.mark_attendance, name='mark_attendance'),
    
    path('start/<int:unit_id>/', views.start_attendance_session, name='start_attendance'),
    path('end/<int:session_id>/', views.end_attendance_session, name='end_attendance'),
    path('manage-attendance/<int:unit_id>/', views.manage_attendance_session, name='manage_attendance_session'),
    path('session/<int:session_id>/details/', views.view_session_details, name='view_session_details'),
    path('session/<int:session_id>/export/', views.export_session_attendance, name='export_session_attendance'),
    path('teacher/update-sessions/', update_session_records, name='update_session_records'),
    
]





