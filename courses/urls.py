# courses/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('units/', views.manage_units, name='manage_units'),  # Manage units for admin
    path('units/create/', views.create_unit, name='create_unit'),  # Create unit
    path('units/<int:unit_id>/edit/', views.edit_unit, name='edit_unit'),  # Edit unit
    path('units/<int:unit_id>/delete/', views.delete_unit, name='delete_unit'),  # Delete unit
    path('enroll/', views.enroll_unit, name='enroll_unit'),  # Enroll in unit for students
    path('enrolled_units/', views.enrolled_units, name='enrolled_units'),
    path('units/<int:unit_id>/withdraw/', views.withdraw_unit, name='withdraw_unit'),  # Withdraw from unit
    path('assign_unit/', views.assign_unit, name='assign_unit'),  # Assign unit to teacher
]
