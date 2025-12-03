from django.db import models
from accounts.models import Student, CustomUser
from courses.models import Unit
from django.utils import timezone

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    unit = models.ForeignKey(Unit, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    status = models.CharField(max_length=10, choices=[('present', 'Present'), ('absent', 'Absent')])
    session = models.ForeignKey('AttendanceSession', on_delete=models.CASCADE, null=True)

    class Meta:
        unique_together = ['student', 'session']  # Ensure one attendance per

    def __str__(self):
        return f'{self.student.user.username} - {self.unit.name} - {self.status}'

# models.py in attendance app

from django.db import models
from django.utils import timezone
from courses.models import Unit
from accounts.models import CustomUser

class AttendanceSession(models.Model):
    unit = models.ForeignKey(Unit, on_delete=models.CASCADE)
    teacher = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    
    ended_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)  # Status of the session (active or closed)
    created_at = models.DateTimeField(default=timezone.now)  # Timestamp when the session was created

    def __str__(self):
        # Get the status of the session (Active or Closed)
        status = "Active" if self.is_active else "Closed"
        # Return a string representation including the unit name, teacher, status, and start time
        return f"{self.unit.name} | {self.teacher.username} | {status} | Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M')}"

    def get_attendance_count(self):
        return self.attendancerecord_set.count()

    def get_attendance_percentage(self):
        total_students = self.unit.get_student_count()
        if total_students > 0:
            attended = self.get_attendance_count()
            return (attended / total_students) * 100
        return 0
