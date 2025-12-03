from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponseForbidden
from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.db import transaction
from .models import Unit
from .forms import UnitForm
from accounts.models import Student

def is_admin(user):
    """Check if user is an admin"""
    return user.is_authenticated and user.role == 'admin'

def is_student(user):
    """Check if user is a student"""
    return user.is_authenticated and user.role == 'student'

@login_required
@user_passes_test(is_admin)
def manage_units(request):
    """View for administrators to manage all units"""
    try:
        units = Unit.objects.all().select_related('teacher')
        return render(request, 'courses/unit_list.html', {
            'units': units,
            'title': 'Manage Units'
        })
    except Exception as e:
        messages.error(request, f"Error loading units: {str(e)}")
        return redirect('admin_dashboard')

@login_required
@user_passes_test(is_admin)
def create_unit(request):
    """View for administrators to create new units"""
    try:
        if request.method == 'POST':
            form = UnitForm(request.POST)
            if form.is_valid():
                unit = form.save()
                messages.success(request, f"Unit '{unit.name}' created successfully.")
                return redirect('manage_units')
            else:
                messages.error(request, "Please correct the errors below.")
        else:
            form = UnitForm()

        return render(request, 'courses/create_unit.html', {
            'form': form,
            'title': 'Create New Unit'
        })
    except Exception as e:
        messages.error(request, f"Error creating unit: {str(e)}")
        return redirect('manage_units')

@login_required
@user_passes_test(is_admin)
def edit_unit(request, unit_id):
    """View for administrators to edit existing units"""
    try:
        unit = get_object_or_404(Unit, id=unit_id)

        if request.method == 'POST':
            form = UnitForm(request.POST, instance=unit)
            if form.is_valid():
                updated_unit = form.save()
                messages.success(request, f"Unit '{updated_unit.name}' updated successfully.")
                return redirect('manage_units')
            else:
                messages.error(request, "Please correct the errors below.")
        else:
            form = UnitForm(instance=unit)

        return render(request, 'courses/edit_unit.html', {
            'form': form,
            'unit': unit,
            'title': f'Edit Unit: {unit.name}'
        })
    except Exception as e:
        messages.error(request, f"Error updating unit: {str(e)}")
        return redirect('manage_units')

@login_required
@user_passes_test(is_admin)
def delete_unit(request, unit_id):
    """View for administrators to delete units"""
    try:
        unit = get_object_or_404(Unit, id=unit_id)

        if request.method == 'POST':
            unit_name = unit.name
            unit.delete()
            messages.success(request, f"Unit '{unit_name}' deleted successfully.")
            return redirect('manage_units')

        return render(request, 'courses/confirm_delete_unit.html', {
            'unit': unit,
            'title': f'Delete Unit: {unit.name}'
        })
    except Exception as e:
        messages.error(request, f"Error deleting unit: {str(e)}")
        return redirect('manage_units')

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.db import transaction
from django.utils import timezone
from .models import Unit, UnitEnrollment
from accounts.models import Student
from django.contrib.auth.decorators import login_required, user_passes_test

def is_student(user):
    """Check if user is a student"""
    return user.is_authenticated and user.role == 'student'

@login_required
@user_passes_test(is_student)
def enroll_unit(request):
    """View for students to enroll in units"""
    student = get_object_or_404(Student, user=request.user)
    
    # Get units the student is not enrolled in or where enrollment is inactive
    available_units = Unit.objects.exclude(
        unitenrollment__student=student,
        unitenrollment__is_active=True
    ).select_related('teacher')

    # POST request: Handle form submission for enrolling a unit
    if request.method == 'POST':
        unit_id = request.POST.get('unit_id')  # Get selected unit ID from the form
        
        if not unit_id:
            messages.error(request, "Please select a unit to enroll in.")
            return redirect('enroll_unit')  # Redirect back if no unit selected
        
        try:
            # Transaction management to ensure atomicity
            with transaction.atomic():
                # Fetch the unit by ID
                unit = Unit.objects.get(id=unit_id)
                
                # Check if the student is already enrolled and the enrollment is active
                enrollment, created = UnitEnrollment.objects.get_or_create(
                    unit=unit,
                    student=student,
                    defaults={'is_active': True, 'enrolled_at': timezone.now()}
                )
                
                if created:
                    messages.success(request, f"Successfully enrolled in {unit.name}.")
                elif not enrollment.is_active:
                    # Reactivate the enrollment if it was previously inactive
                    enrollment.is_active = True
                    enrollment.enrolled_at = timezone.now()
                    enrollment.save()
                    messages.success(request, f"Successfully re-enrolled in {unit.name}.")
                else:
                    # If already enrolled and active
                    messages.warning(request, f"You are already enrolled in {unit.name}.")
                
                # Redirect to the student profile after successful enrollment
                return redirect('student_profile')
        
        except Unit.DoesNotExist:
            messages.error(request, "The selected unit does not exist.")
            return redirect('enroll_unit')
        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
            return redirect('student_profile')  # Redirect to profile on error

    # Render the available units for enrollment
    return render(request, 'courses/enroll_unit.html', {
        'units': available_units,
        'title': 'Enroll in Units'
    })



@login_required
@user_passes_test(is_student)
def withdraw_unit(request, unit_id):
    """View for students to withdraw from units"""
    try:
        student = get_object_or_404(Student, user=request.user)
        enrollment = get_object_or_404(
            UnitEnrollment,
            unit_id=unit_id,
            student=student,
            is_active=True  # Ensure the student is currently enrolled in the unit
        )

        if request.method == 'POST':
            enrollment.is_active = False  # Mark enrollment as inactive
            enrollment.save()
            messages.success(request, f"Successfully withdrawn from {enrollment.unit.name}.")
            return redirect('student_profile')

        return render(request, 'courses/confirm_withdraw.html', {
            'enrollment': enrollment
        })

    except Exception as e:
        messages.error(request, f"Error during withdrawal: {str(e)}")
        return redirect('student_profile')

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from .models import Unit
from accounts.models import CustomUser

def is_teacher(user):
    """Check if user is a teacher"""
    return user.is_authenticated and user.role == 'teacher'

@login_required
@user_passes_test(is_teacher)

def assign_unit(request):
    """Allow a teacher to select a unit to teach"""#-
    """#+
    Allow a teacher to select a unit to teach.#+
#+
    This view function is designed for teachers to assign themselves to teach a specific unit.#+
    It retrieves all units that don't have a teacher assigned yet (or units not assigned to the current teacher)#+
    and presents them to the teacher for selection. Upon form submission, the selected unit is assigned to the teacher.#+
#+
    Parameters:#+
    request (HttpRequest): The request object containing the user's session and POST data.#+
#+
    Returns:#+
    HttpResponseRedirect: If the request method is POST and the unit assignment is successful,#+
    redirect to the teacher's dashboard. Otherwise, render the 'courses/assign_unit.html' template with the list of units.#+
    """#+
    teacher = request.user  # The logged-in teacher

    # Get all units that don't have a teacher assigned yet (or units not assigned to this teacher)
    units_without_teacher = Unit.objects.filter(teacher__isnull=True)

    if request.method == 'POST':
        unit_id = request.POST.get('unit_id')
        if not unit_id:
            messages.error(request, "No unit selected.")
            return redirect('assign_unit')

        try:
            unit = get_object_or_404(Unit, id=unit_id)
            if unit.teacher is not None:
                messages.error(request, "This unit is already assigned to another teacher.")
                return redirect('assign_unit')

            # Assign the selected unit to the teacher
            unit.teacher = teacher
            unit.save()

            messages.success(request, f"You have successfully been assigned to teach '{unit.name}'.")
            return redirect('teacher_dashboard')  # Redirect to the teacher's dashboard or any relevant page

        except Unit.DoesNotExist:
            messages.error(request, "Selected unit does not exist.")
            return redirect('assign_unit')

    return render(request, 'courses/assign_unit.html', {
        'units': units_without_teacher,  # Only units without a teacher
        'title': 'Assign Unit to Teach'
    })

import logging
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from .models import Unit, UnitEnrollment
from accounts.models import Student
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.exceptions import PermissionDenied

# Define the logger
logger = logging.getLogger(__name__)

import logging
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from .models import Unit, UnitEnrollment
from accounts.models import Student
from django.contrib.auth.decorators import login_required, user_passes_test

# Define the logger
logger = logging.getLogger(__name__)

def is_student(user):
    """Check if user is a student"""
    return user.is_authenticated and hasattr(user, 'role') and user.role == 'student'

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.db.models import Q
import logging
from .models import Student, Unit, UnitEnrollment

logger = logging.getLogger(__name__)

@login_required
@user_passes_test(is_student, login_url='login')
def enrolled_units(request):
    """View to display student's enrolled units"""
    try:
        # Log that the view has been accessed
        logger.info(f"Student {request.user.username} is accessing their enrolled units page.")

        # Get the student object using the authenticated user
        student = Student.objects.filter(user=request.user).first()

        if not student:
            logger.error(f"No student profile found for user {request.user.username}")
            messages.error(request, "Student profile not found. Please contact administrator.")
            return redirect('student_profile')

        # Log the student being retrieved
        logger.info(f"Student {student.user.username} found with ID {student.id}.")

        # Fetch active enrollments for the student
        enrollments = UnitEnrollment.objects.filter(
            student=student,
            is_active=True
        ).select_related('unit', 'unit__teacher')

        # Get the units from enrollments
        enrolled_units = [enrollment.unit for enrollment in enrollments]

        # Log how many units were found
        logger.info(f"Found {len(enrolled_units)} units for student {student.user.username}.")

        if not enrolled_units:
            logger.warning(f"No units found for student {student.user.username}.")
            messages.info(request, "You are not enrolled in any units yet.")

        # Add active session information to units
        for unit in enrolled_units:
            unit.has_active_session = hasattr(unit, 'attendancesession_set') and \
                                    unit.attendancesession_set.filter(is_active=True).exists()

        # Prepare context for rendering
        context = {
            'enrolled_units': enrolled_units,
            'student': student,
        }

        return render(request, 'courses/enrolled_units.html', context)

    except Exception as error_message:
        # Log the detailed error
        logger.exception(f"Detailed error in enrolled_units view for {request.user.username}")
        logger.error(f"Error message: {str(error_message)}")

        # Show an error message to the user
        messages.error(request, "An error occurred while fetching your enrolled units.")

        # Redirect to the student profile
        return redirect('student_profile')