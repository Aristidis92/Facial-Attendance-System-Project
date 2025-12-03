# attendance/views.py
import csv
from django.shortcuts import render
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from face_recognition.recognition import get_face_embedding
from face_recognition.detection import detect_face
from attendance.models import Attendance
from courses.models import Unit,UnitEnrollment
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from django.http import JsonResponse
from django.shortcuts import render
from .models import Unit, Attendance, AttendanceSession
from django.http import HttpResponseForbidden
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
import base64
import logging
import os
import json

logger = logging.getLogger(__name__)

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .models import Unit, Attendance, AttendanceSession, Student
from face_recognition.recognition import get_face_embedding
from face_recognition.detection import detect_face
from face_recognition.utils import decode_base64_image

logger = logging.getLogger(__name__)

@login_required
def mark_attendance(request, unit_id):
    try:
        # Get the unit and verify active session
        unit = get_object_or_404(Unit, id=unit_id)
        active_session = AttendanceSession.objects.filter(unit=unit, is_active=True).first()

        if request.method == 'GET':
            return render(request, 'attendance/mark.html', {
                'unit': unit,
                'unit_id': unit_id,
                'has_active_session': bool(active_session)
            })

        if not active_session:
            logger.warning(f"No active session found for unit {unit_id}")
            return JsonResponse({
                'status': 'No active attendance session for this unit'
            }, status=403)

        if request.method == 'POST':
            logger.info(f"Processing attendance for unit {unit_id}")

            # Get the face data
            face_data = request.POST.get('face_data')
            if not face_data:
                logger.error("No face data received in request")
                return JsonResponse({
                    'status': 'No image data received'
                }, status=400)

            try:
                # Ensure temp directory exists
                temp_dir = os.path.join('media', 'temp')
                os.makedirs(temp_dir, exist_ok=True)

                # Save the image
                filename = f"temp_{request.user.id}_{unit_id}.jpg"
                filepath = os.path.join(temp_dir, filename)

                try:
                    # Process and save the image
                    if ',' in face_data:
                        face_data = face_data.split(',')[1]

                    # Decode and save image
                    image_data = base64.b64decode(face_data)
                    with open(filepath, 'wb') as f:
                        f.write(image_data)

                    logger.info(f"Image saved successfully to {filepath}")

                except Exception as e:
                    logger.error(f"Error saving image: {str(e)}")
                    return JsonResponse({
                        'status': 'Error saving image data'
                    }, status=400)

                try:
                    # Detect face
                    logger.info("Attempting face detection")
                    face_tensor = detect_face(filepath)

                    if face_tensor is None:
                        logger.warning("No face detected in image")
                        return JsonResponse({
                            'status': 'No face detected. Please ensure your face is clearly visible and try again.'
                        })

                    # Perform liveness detection
                    logger.info("Performing liveness check")
                    from face_recognition.liveness_model import is_live_face  # Updated import

                    if not is_live_face(face_tensor):
                        logger.warning("Liveness check failed")
                        return JsonResponse({
                            'status': 'Liveness check failed. Please ensure you are using a live camera feed.'
                        })

                    logger.info("Liveness check passed")

                    # Get face embedding
                    logger.info("Getting face embedding")
                    new_embedding = get_face_embedding(face_tensor)

                    if new_embedding is None:
                        logger.error("Failed to generate face embedding")
                        return JsonResponse({
                            'status': 'Failed to process face features. Please try again.'
                        })

                    # Get the student making the request
                    student = get_object_or_404(Student, user=request.user)

                    # Verify student is enrolled in the unit
                    if not student.unitenrollment_set.filter(unit=unit, is_active=True).exists():
                        logger.warning(f"Student {student.id} not enrolled in unit {unit_id}")
                        return JsonResponse({
                            'status': 'You are not enrolled in this unit'
                        }, status=403)

                    # Check if attendance already marked
                    if Attendance.objects.filter(student=student, session=active_session).exists():
                        logger.info(f"Attendance already marked for student {student.id}")
                        return JsonResponse({
                            'status': 'Your attendance has already been marked for this session'
                        })

                    # Verify face match with stored embedding
                    if not student.embedding_path:
                        logger.error(f"No stored embedding for student {student.id}")
                        return JsonResponse({
                            'status': 'No stored face data found. Please update your profile.'
                        })

                    stored_emb_path = os.path.join('media', student.embedding_path)
                    if not os.path.exists(stored_emb_path):
                        logger.error(f"Embedding file not found for student {student.id}")
                        return JsonResponse({
                            'status': 'Face data file not found. Please update your profile.'
                        })

                    # Load stored embedding and compare
                    try:
                        stored_emb = np.load(stored_emb_path)
                        similarity = cosine_similarity([new_embedding], [stored_emb])[0][0]
                        logger.info(f"Face similarity score: {similarity}")

                        if similarity > 0.8:  # Threshold for face match
                            # Mark attendance
                            Attendance.objects.create(
                                student=student,
                                session=active_session,
                                unit=unit,
                                status='present'
                            )
                            logger.info(f"Attendance marked successfully for student {student.id}")
                            return JsonResponse({
                                'status': 'Attendance marked successfully!'
                            })
                        else:
                            logger.warning(f"Face verification failed for student {student.id}")
                            return JsonResponse({
                                'status': 'Face verification failed. Please try again.'
                            })

                    except Exception as e:
                        logger.error(f"Error comparing face embeddings: {str(e)}")
                        return JsonResponse({
                            'status': 'Error verifying face match. Please try again.'
                        })

                except Exception as e:
                    logger.error(f"Error in face recognition process: {str(e)}")
                    return JsonResponse({
                        'status': 'Error processing face recognition. Please try again.'
                    })

            finally:
                # Cleanup: Remove temporary file
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")

    except Exception as e:
        logger.exception("Unexpected error in mark_attendance view")
        return JsonResponse({
            'status': 'An unexpected error occurred. Please try again.'
        }, status=500)
    

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import Attendance
from courses.models import Unit

@login_required
def teacher_dashboard(request):
    teacher_units = Unit.objects.filter(teacher=request.user)
    attendance_records = Attendance.objects.filter(unit__in=teacher_units).order_by('-date')

    context = {
        'units': teacher_units,
        'records': attendance_records,
    }
    return render(request, 'attendance/teacher_dashboard.html', context)








from face_recognition.detection import detect_face
from face_recognition.recognition import get_face_embedding

import numpy as np



def recognize_face(image_path):
    import joblib, os

    model_path = "media/models/face_classifier.pkl"
    if not os.path.exists(model_path):
        return None  # Or return an error message like "Model not trained yet"

    clf = joblib.load(model_path)

    face_tensor = detect_face(image_path)
    emb = get_face_embedding(face_tensor)
    if emb is not None:
        prediction = clf.predict([emb])[0]
        probability = clf.predict_proba([emb])[0].max()
        if probability > 0.8:
            return prediction
    return None


from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib import messages
from attendance.models import AttendanceSession
from courses.models import Unit

def is_teacher(user):
    return user.is_authenticated and user.role == 'teacher'

@login_required
@user_passes_test(is_teacher)
def start_attendance_session(request, unit_id):
    unit = get_object_or_404(Unit, id=unit_id, teacher=request.user)

    # Check for existing active session
    active_session = AttendanceSession.objects.filter(
        unit=unit,
        is_active=True
    ).first()

    if active_session:
        messages.warning(
            request,
            f"An active session for {unit.name} already exists (started at {active_session.created_at})"
        )
    else:
        # Create new session
        new_session = AttendanceSession.objects.create(
            unit=unit,
            teacher=request.user
        )
        messages.success(
            request,
            f"Attendance session for {unit.name} started successfully"
        )

    return redirect('teacher_dashboard')
@login_required
@user_passes_test(is_teacher)
def end_attendance_session(request, session_id):
    session = get_object_or_404(
        AttendanceSession,
        id=session_id,
        teacher=request.user,
        is_active=True
    )

    session.is_active = False
    session.ended_at = timezone.now()
    session.save()

    messages.success(
        request,
        f"Attendance session for {session.unit.name} ended successfully"
    )
    return redirect('teacher_dashboard')

from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.decorators import login_required, user_passes_test
from django.utils.timezone import now
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count, Avg, Max, Min
from django.utils import timezone
from datetime import timedelta


@login_required
@user_passes_test(is_teacher)
def manage_attendance_session(request, unit_id):
    # Get the unit
    unit = get_object_or_404(Unit, id=unit_id, teacher=request.user)

    # Get active session
    active_session = AttendanceSession.objects.filter(unit=unit, is_active=True).first()

    # Get enrolled students through UnitEnrollment
    enrolled_students = UnitEnrollment.objects.filter(
        unit=unit,
        is_active=True
    ).select_related('student', 'student__user').order_by('student__student_id')

    enrolled_count = enrolled_students.count()

    # Calculate attendance statistics if there's an active session
    attendance = None
    if active_session:
        present_count = Attendance.objects.filter(
            session=active_session,
            status='present'
        ).count()

        attendance = {
            'present': present_count,
            'total': enrolled_count,
            'percentage': round((present_count / enrolled_count * 100) if enrolled_count > 0 else 0, 1)
        }

    # Get attendance history
    filter_param = request.GET.get('filter', 'all')
    attendance_queryset = AttendanceSession.objects.filter(unit=unit)

    if filter_param == 'today':
        attendance_queryset = attendance_queryset.filter(
            created_at__date=timezone.now().date()
        )
    elif filter_param == 'week':
        attendance_queryset = attendance_queryset.filter(
            created_at__gte=timezone.now() - timedelta(days=7)
        )
    elif filter_param == 'month':
        attendance_queryset = attendance_queryset.filter(
            created_at__gte=timezone.now() - timedelta(days=30)
        )

    # Calculate overall attendance statistics
    attendance_stats = attendance_queryset.aggregate(
        total_sessions=Count('id'),
    )

    # Calculate average, highest, and lowest attendance
    total_sessions = attendance_stats['total_sessions']
    attendance_percentages = []

    for session in attendance_queryset:
        present_count = Attendance.objects.filter(
            session=session,
            status='present'
        ).count()
        if enrolled_count > 0:
            percentage = (present_count / enrolled_count) * 100
            attendance_percentages.append(percentage)

    average_attendance = round(sum(attendance_percentages) / len(attendance_percentages), 1) if attendance_percentages else 0
    highest_attendance = round(max(attendance_percentages, default=0), 1)
    lowest_attendance = round(min(attendance_percentages, default=0), 1)

    # Paginate attendance history
    paginator = Paginator(attendance_queryset.order_by('-created_at'), 10)
    page = request.GET.get('page')
    attendance_history = paginator.get_page(page)

    # Calculate statistics for each session in the current page
    for session in attendance_history:
        present_count = Attendance.objects.filter(
            session=session,
            status='present'
        ).count()
        session.present_count = present_count
        session.total_students = enrolled_count
        session.attendance_percentage = round(
            (present_count / enrolled_count * 100) if enrolled_count > 0 else 0,
            1
        )

    if request.method == 'POST':
        if 'end_session' in request.POST and active_session:
            try:
                # Get students who already marked attendance
                attended_students = Attendance.objects.filter(
                    session=active_session
                ).values_list('student_id', flat=True)

                # Create bulk absent records for students who didn't mark attendance
                bulk_absent_records = []
                for enrollment in enrolled_students:
                    if enrollment.student.id not in attended_students:
                        bulk_absent_records.append(
                            Attendance(
                                student=enrollment.student,
                                unit=unit,
                                session=active_session,
                                status='absent',
                                date=now()
                            )
                        )

                # Bulk create absent records
                if bulk_absent_records:
                    Attendance.objects.bulk_create(bulk_absent_records)

                # End the session
                active_session.is_active = False
                active_session.ended_at = now()
                active_session.save()

                messages.success(
                    request,
                    f"Attendance session ended. Marked {len(bulk_absent_records)} students as absent."
                )
                return redirect('teacher_dashboard')

            except Exception as e:
                messages.error(request, f"Error ending session: {str(e)}")
                return redirect('manage_attendance_session', unit_id=unit.id)

        elif 'start_session' in request.POST and not active_session:
            try:
                # Create new session without started_at (it will use auto_now_add)
                active_session = AttendanceSession.objects.create(
                    unit=unit,
                    teacher=request.user
                )
                messages.success(request, f"Attendance session for {unit.name} has been started.")
                return redirect('manage_attendance_session', unit_id=unit.id)

            except Exception as e:
                messages.error(request, f"Error starting session: {str(e)}")
                return redirect('manage_attendance_session', unit_id=unit.id)

    context = {
        'unit': unit,
        'active_session': active_session,
        'enrolled_students': enrolled_students,
        'enrolled_count': enrolled_count,
        'attendance': attendance,
        'attendance_history': attendance_history,
        'total_sessions': total_sessions,
        'average_attendance': average_attendance,
        'highest_attendance': highest_attendance,
        'lowest_attendance': lowest_attendance,
    }
    return render(request, 'attendance/manage_attendance_session.html', context)

@login_required
@user_passes_test(is_teacher)
def view_session_details(request, session_id):
    # Get the attendance session
    session = get_object_or_404(AttendanceSession, id=session_id)

    # Ensure the teacher owns this session
    if session.unit.teacher != request.user:
        messages.error(request, "You don't have permission to view this session.")
        return redirect('teacher_dashboard')

    # Get all attendance records for this session
    attendance_records = Attendance.objects.filter(
        session=session
    ).select_related('student', 'student__user')

    # Get all enrolled students
    enrolled_students = UnitEnrollment.objects.filter(
        unit=session.unit,
        is_active=True
    ).select_related('student', 'student__user')

    # Calculate statistics
    total_students = enrolled_students.count()
    present_students = attendance_records.filter(status='present').count()
    absent_students = attendance_records.filter(status='absent').count()
    attendance_percentage = round((present_students / total_students * 100) if total_students > 0 else 0, 1)

    context = {
        'session': session,
        'attendance_records': attendance_records,
        'total_students': total_students,
        'present_students': present_students,
        'absent_students': absent_students,
        'attendance_percentage': attendance_percentage,
    }

    return render(request, 'attendance/session_details.html', context)
@login_required
@user_passes_test(is_teacher)
def export_session_attendance(request, session_id):
    session = get_object_or_404(AttendanceSession, id=session_id)

    # Ensure the teacher owns this session
    if session.unit.teacher != request.user:
        messages.error(request, "You don't have permission to export this session's attendance.")
        return redirect('teacher_dashboard')

    # Get attendance records
    attendance_records = Attendance.objects.filter(
        session=session
    ).select_related('student', 'student__user')

    # Create the response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="attendance_{session.unit.code}_{session.created_at.date()}.csv"'

    # Create the CSV writer
    writer = csv.writer(response)
    writer.writerow(['Student ID', 'Name', 'Status', 'Time'])

    # Add the records
    for record in attendance_records:
        writer.writerow([
            record.student.student_id,
            record.student.user.get_full_name(),
            record.status,
            record.date.strftime('%I:%M %p')
        ])

    return response

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from attendance.models import AttendanceSession, Attendance, Unit
from .models import CustomUser

def is_teacher(user):
    return user.role == 'teacher'  # Adjust this based on your CustomUser model

@login_required
def update_session_records(request):
    try:
        # Get the teacher's units
        teacher_units = Unit.objects.filter(teacher=request.user)

        # Get active sessions for these units
        active_sessions = AttendanceSession.objects.filter(
            unit__in=teacher_units,
            is_active=True
        ).select_related('unit')

        if request.method == 'POST':
            # Debugging: Print the POST data
            print("POST data:", request.POST)

            session_id = request.POST.get('session_id')
            student_id = request.POST.get('student_id')
            status = request.POST.get('status')

            # Debugging: Print the extracted values
            print(f"Session ID: {session_id}, Student ID: {student_id}, Status: {status}")

            if not all([session_id, student_id, status]):
                messages.error(request, "Missing required information")
                return redirect('update_session_records')

            try:
                session = get_object_or_404(AttendanceSession, id=session_id)
                student = get_object_or_404(Student, id=student_id)

                # Verify that the teacher has access to this unit
                if session.unit not in teacher_units:
                    messages.error(request, "You don't have permission to modify this session")
                    return redirect('update_session_records')

                # Update or create attendance record
                attendance, created = Attendance.objects.update_or_create(
                    student=student,
                    session=session,
                    defaults={
                        'unit': session.unit,
                        'status': status,
                        'date': timezone.now().date()
                    }
                )

                messages.success(
                    request,
                    f"Attendance marked as {status} for {student.user.get_full_name()}"
                )

            except Exception as e:
                messages.error(request, f"Error updating attendance: {str(e)}")

        # Prepare data for template
        sessions_data = []
        for session in active_sessions:
            # Get all enrolled students for this unit
            enrolled_students = session.unit.unitenrollment_set.filter(
                is_active=True
            ).select_related('student', 'student__user')

            # Get existing attendance records for this session
            existing_attendance = dict(
                Attendance.objects.filter(session=session)
                .values_list('student_id', 'status')
            )

            # Prepare student data
            students_data = []
            for enrollment in enrolled_students:
                student = enrollment.student
                status = existing_attendance.get(student.id, 'absent')

                students_data.append({
                    'student': student,
                    'status': status,
                    'marked_present': status == 'present'
                })

            sessions_data.append({
                'session': session,
                'students': students_data,
                'total_students': len(students_data),
                'present_count': sum(1 for s in students_data if s['marked_present'])
            })

        context = {
            'sessions_data': sessions_data,
            'active_sessions_count': len(sessions_data)
        }

        return render(request, 'attendance/update_session_records.html', context)

    except Exception as e:
        messages.error(request, f"An error occurred: {str(e)}")
        return redirect('teacher_dashboard')