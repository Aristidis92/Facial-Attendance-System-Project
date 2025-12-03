from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from courses.models import Unit, UnitEnrollment
from accounts.models import Student
from unittest.mock import patch
from datetime import datetime

User = get_user_model()

class UnitViewsTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user(username='admin', password='adminpass', role='admin')
        self.teacher = User.objects.create_user(username='teacher', password='teachpass', role='teacher')
        self.student_user = User.objects.create_user(username='student', password='studpass', role='student')
        self.student = Student.objects.create(user=self.student_user, student_id='STU001')
        self.unit = Unit.objects.create(name='Mathematics', code='MATH101')

    def test_manage_units_admin_access(self):
        self.client.login(username='admin', password='adminpass')
        response = self.client.get(reverse('manage_units'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Mathematics')

    def test_create_unit_view(self):
        self.client.login(username='admin', password='adminpass')
        response = self.client.post(reverse('create_unit'), {
            'name': 'Physics',
            'code': 'PHY101'
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Unit.objects.filter(code='PHY101').exists())

    def test_edit_unit_view(self):
        self.client.login(username='admin', password='adminpass')
        url = reverse('edit_unit', kwargs={'unit_id': self.unit.id})
        response = self.client.post(url, {'name': 'Updated Name', 'code': 'MATH101'})
        self.assertRedirects(response, reverse('manage_units'))
        self.unit.refresh_from_db()
        self.assertEqual(self.unit.name, 'Updated Name')

    def test_delete_unit_view(self):
        self.client.login(username='admin', password='adminpass')
        url = reverse('delete_unit', kwargs={'unit_id': self.unit.id})
        response = self.client.post(url)
        self.assertRedirects(response, reverse('manage_units'))
        self.assertFalse(Unit.objects.filter(id=self.unit.id).exists())

    def test_enroll_unit_view(self):
        self.client.login(username='student', password='studpass')
        url = reverse('enroll_unit')
        response = self.client.post(url, {'unit_id': self.unit.id})
        self.assertRedirects(response, reverse('student_profile'))
        self.assertTrue(UnitEnrollment.objects.filter(student=self.student, unit=self.unit, is_active=True).exists())

    def test_withdraw_unit_view(self):
        UnitEnrollment.objects.create(student=self.student, unit=self.unit, is_active=True)
        self.client.login(username='student', password='studpass')
        url = reverse('withdraw_unit', kwargs={'unit_id': self.unit.id})
        response = self.client.post(url)
        self.assertRedirects(response, reverse('student_profile'))
        enrollment = UnitEnrollment.objects.get(student=self.student, unit=self.unit)
        self.assertFalse(enrollment.is_active)

    def test_assign_unit_view(self):
        self.client.login(username='teacher', password='teachpass')
        url = reverse('assign_unit')
        response = self.client.post(url, {'unit_id': self.unit.id})
        self.assertRedirects(response, reverse('teacher_dashboard'))
        self.unit.refresh_from_db()
        self.assertEqual(self.unit.teacher, self.teacher)

    def test_enrolled_units_page(self):
        UnitEnrollment.objects.create(student=self.student, unit=self.unit, is_active=True)
        self.client.login(username='student', password='studpass')
        url = reverse('enrolled_units')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Mathematics')
