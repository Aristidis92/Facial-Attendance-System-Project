from django.db import models
from accounts.models import CustomUser, Student
from django.utils.timezone import now


class Unit(models.Model):
    code = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    teacher = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    students = models.ManyToManyField(
        Student,
        through='UnitEnrollment',
        through_fields=('unit', 'student'),
        blank=True
    )
    created_at = models.DateTimeField(default=now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['code']


    def __str__(self):
        """#+
        Returns a string representation of the Unit object.#+
#+
        The string representation is a combination of the unit's code and name,#+
        separated by a hyphen.#+
#+
        Parameters:#+
        None#+
#+
        Returns:#+
        str: A string in the format "{code} - {name}".#+
        """#+
        return f"{self.code} - {self.name}"
    def has_active_session(self):
        from attendance.models import AttendanceSession
        return AttendanceSession.objects.filter(unit=self, is_active=True).exists()

class UnitEnrollment(models.Model):
    unit = models.ForeignKey(Unit, on_delete=models.CASCADE)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    enrolled_at = models.DateTimeField(default=now)
    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = ['unit', 'student']
        ordering = ['-enrolled_at']

    def __str__(self):
        return f"{self.student} enrolled in {self.unit}"

    def has_active_session(self):
        return self.attendancesession_set.filter(is_active=True).exists()

    def get_active_session(self):
        return self.attendancesession_set.filter(is_active=True).first()

    def get_total_attendance(self):
        return self.attendancesession_set.aggregate(
            total=Count('attendancerecord')
        )['total']

    def get_student_count(self):
        return self.students.count()