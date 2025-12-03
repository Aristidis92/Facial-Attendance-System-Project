
import uuid
import numpy as np
import os
from django.shortcuts import render, redirect, get_object_or_404
from .forms import StudentRegistrationForm
from .models import Student
from face_recognition.utils import decode_base64_image
from face_recognition.detection import detect_face
from face_recognition.recognition import get_face_embedding
from datetime import date
from accounts.models import CustomUser, Student
from django.core.paginator import Paginator
from django.contrib.auth import get_user_model
from django.core.paginator import Paginator
from django.db.models import Count, Exists, OuterRef
from face_recognition.liveness import is_live_face
import numpy as np
from django.shortcuts import render, redirect
from django.contrib import messages
from django.db import transaction
import logging
from .forms import StudentRegistrationForm
from .models import Student
from face_recognition.utils import decode_base64_image


# Configure logging
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_DIMENSION = 512
FACE_SIMILARITY_THRESHOLD = 0.6
MEDIA_ROOT = 'media'
FACES_DIR = os.path.join(MEDIA_ROOT, 'faces')
EMBEDDINGS_DIR = os.path.join(MEDIA_ROOT, 'embeddings')

def validate_embedding(embedding, expected_dim=EMBEDDING_DIMENSION):
    """Validate the face embedding"""
    if not isinstance(embedding, np.ndarray):
        return False, "Embedding is not a numpy array"
    if embedding.ndim != 1:
        return False, f"Expected 1D array, got {embedding.ndim}D"
    if embedding.shape[0] != expected_dim:
        return False, f"Expected dimension {expected_dim}, got {embedding.shape[0]}"
    if np.isnan(embedding).any():
        return False, "Embedding contains NaN values"
    return True, "Embedding is valid"

def cleanup_registration(image_path=None, user_obj=None):
    """Clean up files and user object on registration failure"""
    try:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            logger.debug(f"Removed temporary image: {image_path}")
    except Exception as e:
        logger.error(f"Error cleaning up image {image_path}: {e}")

    try:
        if user_obj:
            user_obj.delete()
            logger.debug(f"Deleted user: {user_obj.username}")
    except Exception as e:
        logger.error(f"Error deleting user {user_obj}: {e}")

@transaction.atomic
def student_register(request):
    """Handle student registration with face recognition"""

    # Ensure required directories exist
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    if request.method == 'POST':
        logger.info("🎯 Processing student registration POST request")
        form = StudentRegistrationForm(request.POST)

        if form.is_valid():
            try:
                # Create user and get face data
                user = form.save()
                face_data = form.cleaned_data['face_data']
                logger.debug(f"Processing registration for user: {user.username}")

                # Get student object
                student = Student.objects.get(user=user)

                # Process face image
                try:
                    filename = f"{uuid.uuid4()}.jpg"
                    image_path = os.path.join(FACES_DIR, filename)
                    decode_base64_image(face_data, image_path)
                    student.face_image = os.path.join('faces', filename)
                    logger.debug(f"Face image saved: {image_path}")
                except Exception as e:
                    logger.error(f"Failed to save face image: {e}")
                    cleanup_registration(user_obj=user)
                    messages.error(request, "Failed to process face image. Please try again.")
                    return render(request, 'accounts/student_register.html', {'form': form})

                # Detect face
                try:
                    face_tensor = detect_face(image_path)
                    if face_tensor is None:
                        logger.warning("No face detected in uploaded image")
                        cleanup_registration(image_path, user)
                        messages.error(request, "No face detected. Please try again with a clear face photo.")
                        return render(request, 'accounts/student_register.html', {'form': form})
                    logger.debug("Face detected successfully")
                except Exception as e:
                    logger.error(f"Error during face detection: {e}")
                    cleanup_registration(image_path, user)
                    messages.error(request, "Error detecting face. Please try again.")
                    return render(request, 'accounts/student_register.html', {'form': form})

                # Perform liveness check
                try:
                    result = is_live_face(face_tensor)
                    if isinstance(result, tuple):
                        is_live, confidence = result
                    else:
                        is_live = result
                        confidence = None

                    if not is_live:
                        logger.warning(f"Liveness check failed. Confidence: {confidence}")
                        cleanup_registration(image_path, user)
                        messages.error(request, "Liveness check failed. Please try again using your live face.")
                        return render(request, 'accounts/student_register.html', {'form': form})

                    logger.info(f"Liveness check passed. Confidence: {confidence}")
                except Exception as e:
                    logger.error(f"Error during liveness check: {e}")
                    cleanup_registration(image_path, user)
                    messages.error(request, "Error during liveness check. Please try again.")
                    return render(request, 'accounts/student_register.html', {'form': form})

                # Generate face embedding
                try:
                    embedding = get_face_embedding(face_tensor)
                    valid, message = validate_embedding(embedding)
                    if not valid:
                        logger.error(f"Invalid embedding: {message}")
                        cleanup_registration(image_path, user)
                        messages.error(request, "Error processing face features. Please try again.")
                        return render(request, 'accounts/student_register.html', {'form': form})
                    logger.debug("Face embedding generated successfully")
                except Exception as e:
                    logger.error(f"Error generating face embedding: {e}")
                    cleanup_registration(image_path, user)
                    messages.error(request, "Error processing face features. Please try again.")
                    return render(request, 'accounts/student_register.html', {'form': form})

                # Check for duplicate faces
                try:
                    logger.debug("Starting duplicate face check...")
                    existing_students = Student.objects.exclude(embedding_path__isnull=True)

                    for other in existing_students:
                        try:
                            other_path = os.path.join(MEDIA_ROOT, other.embedding_path)
                            if not os.path.exists(other_path):
                                logger.warning(f"Embedding file not found: {other_path}")
                                continue

                            existing_embedding = np.load(other_path)
                            distance = np.linalg.norm(embedding - existing_embedding)
                            logger.debug(f"Face distance with student {other.student_id}: {distance:.4f}")

                            if distance < FACE_SIMILARITY_THRESHOLD:
                                logger.warning(f"Duplicate face detected! Distance: {distance:.4f}")
                                cleanup_registration(image_path, user)
                                messages.error(
                                    request,
                                    f"This face matches with existing student (ID: {other.student_id}). "
                                    f"Distance: {distance:.4f}"
                                )
                                return render(request, 'accounts/student_register.html', {'form': form})
                        except Exception as e:
                            logger.error(f"Error comparing with student {other.student_id}: {e}")
                            continue

                    logger.debug("No duplicate faces found")
                except Exception as e:
                    logger.error(f"Error during duplicate check: {e}")
                    cleanup_registration(image_path, user)
                    messages.error(request, "Error checking for duplicate faces. Please try again.")
                    return render(request, 'accounts/student_register.html', {'form': form})

                # Save embedding and complete registration
                try:
                    embedding_filename = f"{student.student_id}.npy"
                    embedding_path = os.path.join(EMBEDDINGS_DIR, embedding_filename)
                    np.save(embedding_path, embedding)

                    student.embedding_path = os.path.join('embeddings', embedding_filename)
                    student.save()

                    logger.info(f"✅ Student registered successfully: {student.student_id}")
                    messages.success(request, "Registration successful! Please log in.")
                    return redirect('login')
                except Exception as e:
                    logger.error(f"Error saving embedding: {e}")
                    cleanup_registration(image_path, user)
                    messages.error(request, "Error saving face data. Please try again.")
                    return render(request, 'accounts/student_register.html', {'form': form})

            except Exception as e:
                logger.error(f"Unexpected error during registration: {e}")
                cleanup_registration(
                    image_path if 'image_path' in locals() else None,
                    user if 'user' in locals() else None
                )
                messages.error(request, "An unexpected error occurred. Please try again.")
                return render(request, 'accounts/student_register.html', {'form': form})
        else:
            logger.warning(f"Form validation failed: {form.errors}")
            messages.error(request, "Please correct the errors below.")
    else:
        form = StudentRegistrationForm()
        logger.debug("Displaying empty registration form")

    return render(request, 'accounts/student_register.html', {'form': form})


from .forms import TeacherRegistrationForm, AdminRegistrationForm

def teacher_register(request):
    if request.method == 'POST':
        form = TeacherRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = TeacherRegistrationForm()
    return render(request, 'accounts/teacher_register.html', {'form': form})

from .forms import AdminRegistrationForm  # ✅ Needed at the top

def admin_register(request):
    if request.method == 'POST':
        form = AdminRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Auto-login after registration (optional)
            from django.contrib.auth import login
            login(request, user)

            # Option 1: Redirect to Django Admin Panel
            return redirect('/admin/')

            # Option 2: Redirect to custom dashboard
            # return redirect('admin_dashboard')
    else:
        form = AdminRegistrationForm()
    return render(request, 'accounts/admin_register.html', {'form': form})

from django.contrib.auth.decorators import login_required
from attendance.models import Attendance
from collections import Counter
from django.http import HttpResponseForbidden
from attendance.models import Attendance
from .models import CustomUser

@login_required
def student_profile(request):
    try:
        student = request.user.student
    except Student.DoesNotExist:
        return HttpResponseForbidden("🚫 This profile is only accessible to students.")

    student = request.user.student
    # Get all units the student is enrolled in
    enrolled_units = Unit.objects.filter(unitenrollment__student=student, unitenrollment__is_active=True)

    # Get attendance records for each unit
    attendance_data = []
    for unit in enrolled_units:
        total_sessions = AttendanceSession.objects.filter(unit=unit).count()
        attended_sessions = Attendance.objects.filter(
            student=student,
            unit=unit,
            status='present'
        ).count()

        if total_sessions > 0:
            attendance_percentage = (attended_sessions / total_sessions) * 100
        else:
            attendance_percentage = 0

        attendance_data.append({
            'unit': unit,
            'total_sessions': total_sessions,
            'attended_sessions': attended_sessions,
            'attendance_percentage': round(attendance_percentage, 1)
        })

    # Get recent attendance records
    recent_attendance = Attendance.objects.filter(
        student=student
    ).select_related('unit').order_by('-date')[:10]  # Last 10 attendance records

    context = {
        'student': student,
        'attendance_data': attendance_data,
        'recent_attendance': recent_attendance,
    }
    return render(request, 'accounts/profile.html', context)

from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required

@login_required
def role_redirect(request):
    user = request.user

    if user.role == 'student':
        return redirect('student_profile')
    elif user.role == 'teacher':
        return redirect('teacher_dashboard')  # You can create this view
    elif user.role == 'admin':
        return redirect('admin_dashboard')  # Redirect to Django admin
    else:
        return redirect('/')  # Fallback

@login_required
def teacher_dashboard(request):
    if request.user.role != 'teacher':
        return HttpResponseForbidden("Only teachers can access this page.")

    # You can later add context like assigned courses, attendance stats, etc.
    return render(request, 'accounts/teacher_dashboard.html')

from django.contrib.auth.decorators import login_required
from courses.models import Unit,UnitEnrollment

@login_required
def teacher_dashboard(request):
    units = Unit.objects.filter(teacher=request.user)
    return render(request, 'accounts/teacher_dashboard.html', {'units': units})

import csv
from django.http import HttpResponse

@login_required
def export_attendance_csv(request):
    teacher = request.user
    units = Unit.objects.filter(teacher=teacher)
    records = Attendance.objects.filter(unit__in=units)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="attendance_report.csv"'
    writer = csv.writer(response)
    writer.writerow(['Student', 'Unit', 'Date', 'Status'])

    for record in records:
        writer.writerow([record.student.user.username, record.unit.name, record.date, record.status])

    return response

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
@login_required
@login_required
def admin_dashboard(request):
    if request.user.role != 'admin':
        return HttpResponseForbidden("You are not authorized to access this page.")

    total_users = CustomUser.objects.count()
    total_students = Student.objects.count()
    total_units = Unit.objects.count()
    total_attendance = Attendance.objects.count()

    today = date.today()
    today_attendance = Attendance.objects.filter(date=today)
    status_counts = Counter(record.status for record in today_attendance)

    context = {
        'total_users': total_users,
        'total_students': total_students,
        'total_units': total_units,
        'total_attendance': total_attendance,
        'present_today': status_counts.get('Present', 0),
        'absent_today': status_counts.get('Absent', 0),
    }

    return render(request, 'accounts/admin_dashboard.html', context)

from django.contrib.admin.views.decorators import staff_member_required

@staff_member_required
def admin_dashboard(request):
    return render(request, 'accounts/admin_dashboard.html')

@login_required
def manage_users(request):
    if request.user.role != 'admin':
        return HttpResponseForbidden("Unauthorized")

    role_filter = request.GET.get('role')
    if role_filter in ['student', 'teacher', 'admin']:
        users = CustomUser.objects.filter(role=role_filter)
    else:
        users = CustomUser.objects.all()

    return render(request, 'accounts/manage_users.html', {
        'users': users,
        'selected_role': role_filter
    })

User = get_user_model()

@login_required
def user_list(request):
    if request.user.role != 'admin':
        return HttpResponseForbidden("Not authorized.")

    users = User.objects.all().order_by('-date_joined')
    paginator = Paginator(users, 10)  # 10 users per page
    page = request.GET.get('page')
    users_paginated = paginator.get_page(page)

    return render(request, 'accounts/user_list.html', {
        'users': users_paginated
    })

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponseForbidden
from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.db.models import Q
from django.urls import reverse
from .models import CustomUser
from .forms import AdminEditUserForm

def is_admin(user):
    """Check if user is an admin"""
    return user.is_authenticated and user.role == 'admin'

@login_required
@user_passes_test(is_admin, login_url='login')
def edit_user(request, user_id):
    """View for administrators to edit user details"""
    try:
        # Get the user to edit
        target_user = get_object_or_404(CustomUser, id=user_id)

        # Initialize the form
        form = AdminEditUserForm(
            request.POST or None,
            instance=target_user
        )

        if request.method == 'POST':
            if form.is_valid():
                # Save the form
                user = form.save(commit=False)

                # Prevent removing admin status from last admin
                if (target_user.role == 'admin' and
                    form.cleaned_data['role'] != 'admin' and
                    CustomUser.objects.filter(role='admin').count() == 1):
                    messages.error(request, "Cannot remove admin status from the last administrator.")
                    return render(request, 'accounts/edit_user.html', {
                        'form': form,
                        'user_obj': target_user
                    })

                user.save()
                messages.success(request, f"User {user.username} has been updated successfully.")
                return redirect('user_list')
            else:
                messages.error(request, "Please correct the errors below.")

        return render(request, 'accounts/edit_user.html', {
            'form': form,
            'user_obj': target_user
        })

    except Exception as e:
        messages.error(request, f"An error occurred: {str(e)}")
        return redirect('user_list')

@login_required
@user_passes_test(is_admin, login_url='login')
def delete_user(request, user_id):
    """View for administrators to delete users"""
    try:
        # Get the user to delete
        target_user = get_object_or_404(CustomUser, id=user_id)

        # Prevent self-deletion
        if request.user.id == target_user.id:
            messages.error(request, "Administrators cannot delete their own account.")
            return redirect('user_list')

        # Prevent deleting last admin
        if (target_user.role == 'admin' and
            CustomUser.objects.filter(role='admin').count() == 1):
            messages.error(request, "Cannot delete the last administrator account.")
            return redirect('user_list')

        if request.method == 'POST':
            # Store username for confirmation message
            username = target_user.username

            # Delete user
            target_user.delete()

            messages.success(request, f"User {username} has been deleted successfully.")
            return redirect('user_list')

        return render(request, 'accounts/confirm_delete.html', {
            'user_obj': target_user
        })

    except Exception as e:
        messages.error(request, f"An error occurred: {str(e)}")
        return redirect('user_list')

from django.shortcuts import render
from attendance.models import Unit, Student, AttendanceSession
from django.db.models import Count, Exists, OuterRef
from django.shortcuts import render
from django.contrib.auth.decorators import login_required, user_passes_test

@login_required

def teacher_dashboard(request):
    # Get units with active session information in a single query
    units = Unit.objects.filter(teacher=request.user).annotate(
        has_active_session=Exists(
            AttendanceSession.objects.filter(
                unit=OuterRef('pk'),
                is_active=True
            )
        )
    )

    # Get active sessions for all units in a single query
    active_sessions = AttendanceSession.objects.filter(
        unit__in=units,
        is_active=True
    ).select_related('unit')

    # Create a dictionary mapping unit IDs to their active sessions
    active_sessions_dict = {session.unit_id: session for session in active_sessions}

    # Prepare the units data
    active_units = []
    for unit in units:
        active_units.append({
            'unit': unit,
            'has_active_session': unit.has_active_session,
            'active_session': active_sessions_dict.get(unit.id)
        })

    # Get stats in an optimized way
    context = {
        'units': active_units,
        'unit_count': len(units),
        'student_count': Student.objects.count(),
        'attendance_count': AttendanceSession.objects.filter(
            session__unit__in=units
        ).count(),
    }

    return render(request, 'attendance/teacher_dashboard.html', context)

def teacher_dashboard(request):
    # Get units for the logged-in teacher
    units = Unit.objects.filter(teacher=request.user)

    # Get stats
    unit_count = units.count()
    student_count = Student.objects.count()
    attendance_count = AttendanceSession.objects.filter(session__unit__in=units).count()

    # Get active sessions for each unit
    active_units = []
    for unit in units:
        active_session = AttendanceSession.objects.filter(
            unit=unit,
            is_active=True
        ).first()

        active_units.append({
            'unit': unit,
            'has_active_session': bool(active_session),
            'active_session': active_session
        })

    context = {
        'units': active_units,
        'unit_count': unit_count,
        'student_count': student_count,
        'attendance_count': attendance_count,
    }

    return render(request, 'attendance/teacher_dashboard.html', context)

@login_required
def teacher_dashboard(request):
    """View for teacher dashboard showing their enrolled students for each unit."""
    
    # Ensure the logged-in user is a teacher
    if not request.user.role == 'teacher':
        return HttpResponseForbidden("You are not authorized to view this page.")

    # Get the teacher (CustomUser)
    teacher = request.user
    
    # Fetch the units taught by the teacher
    units = Unit.objects.filter(teacher=teacher)
    
    # Prepare a list of tuples (unit, students)
    unit_students = []
    for unit in units:
        # Fetch students enrolled in each unit via the UnitEnrollment model
        enrolled_students = UnitEnrollment.objects.filter(unit=unit, is_active=True)  # Assuming only active enrollments
        students = [enrollment.student for enrollment in enrolled_students]  # Extract the student from each enrollment
        unit_students.append((unit, students))  # Add tuple (unit, students) to the list

    # Pass the units and enrolled students to the template
    context = {
        'unit_students': unit_students,
    }
    
    return render(request, 'teachers/teacher_dashboard.html', context)

from django.shortcuts import render
from django.db.models import Count, F, Q
from django.db.models.functions import TruncDate
from django.utils import timezone
from datetime import timedelta

def admin_dashboard(request):
    # Get basic statistics
    total_units = Unit.objects.count()
    total_teachers = CustomUser.objects.filter(role='teacher').count()
    total_students = Student.objects.count()

    # Get all units with their details
    units = Unit.objects.all().prefetch_related('students', 'teacher')

    # Get recent attendance data (last 7 days)
    seven_days_ago = timezone.now().date() - timedelta(days=7)
    recent_attendance = (
        Attendance.objects
        .filter(date__gte=seven_days_ago)
        .values('unit', 'unit__code', 'unit__name', 'date')
        .annotate(
            present_count=Count('id', filter=Q(status='present')),
            absent_count=Count('id', filter=Q(status='absent')),
            total_count=Count('id'),
            attendance_rate=F('present_count') * 100.0 / F('total_count')
        )
        .order_by('-date')
    )

    context = {
        'total_units': total_units,
        'total_teachers': total_teachers,
        'total_students': total_students,
        'units': units,
        'recent_attendance': recent_attendance,
    }

    return render(request, 'accounts/admin_dashboard.html', context)

def is_student(user):
    """Check if user is a student"""
    return user.is_authenticated and user.role == 'student'

@login_required
@user_passes_test(is_student)
def update_face_data(request):
    student = get_object_or_404(Student, user=request.user)

    if request.method == 'POST':
        face_data = request.POST.get('face_data')
        if not face_data:
            messages.error(request, "No face data received.")
            return redirect('update_face')

        # Save new face image
        filename = f"{uuid.uuid4()}.jpg"
        path = f"media/faces/{filename}"
        decode_base64_image(face_data, path)
        student.face_image = f"faces/{filename}"

        # Detect and embed
        face_tensor = detect_face(path)
        if face_tensor is not None:
            embedding = get_face_embedding(face_tensor)
            embedding_path = f"media/embeddings/{student.student_id}.npy"
            np.save(embedding_path, embedding)

            student.embedding_path = embedding_path.replace("media/", "")
            student.save()

            messages.success(request, "Face data updated successfully.")
        else:
            messages.error(request, "Face not detected. Please try again.")

        return redirect('student_profile')

    return render(request, 'accounts/update_face.html')
