import os
import numpy as np
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.messages import get_messages
from django.contrib.auth import get_user_model
from unittest.mock import patch, MagicMock
from .models import Student, Unit
from attendance.models import AttendanceSession
from courses.models import UnitEnrollment
from accounts.forms import StudentRegistrationForm, TeacherRegistrationForm, AdminRegistrationForm

class TestSetup(TestCase):
    def setUp(self):
        self.client = Client()
        self.User = get_user_model()

        # Create test users
        self.admin_user = self.User.objects.create_user(
            username='admin',
            password='testpass123',
            email='admin@test.com',
            role='admin'
        )

        self.teacher_user = self.User.objects.create_user(
            username='teacher',
            password='testpass123',
            email='teacher@test.com',
            role='teacher'
        )

        self.student_user = self.User.objects.create_user(
            username='student',
            password='testpass123',
            email='student@test.com',
            role='student'
        )

        # Create test student
        self.student = Student.objects.create(
            user=self.student_user,
            student_id='TEST001'
        )

class TestValidateEmbedding(TestCase):
    def test_valid_embedding(self):
        embedding = np.zeros(512)  # Valid embedding
        valid, message = validate_embedding(embedding)
        self.assertTrue(valid)
        self.assertEqual(message, "Embedding is valid")

    def test_invalid_embedding_type(self):
        embedding = [0] * 512  # Invalid type (list instead of np.ndarray)
        valid, message = validate_embedding(embedding)
        self.assertFalse(valid)
        self.assertEqual(message, "Embedding is not a numpy array")

    def test_invalid_embedding_dimension(self):
        embedding = np.zeros((512, 1))  # Invalid dimension
        valid, message = validate_embedding(embedding)
        self.assertFalse(valid)
        self.assertEqual(message, "Expected 1D array, got 2D")

class TestStudentRegistration(TestSetup):
    @patch('face_recognition.detection.detect_face')
    @patch('face_recognition.recognition.get_face_embedding')
    @patch('face_recognition.liveness.is_live_face')
    def test_successful_registration(self, mock_is_live, mock_get_embedding, mock_detect_face):
        # Mock the face recognition functions
        mock_detect_face.return_value = np.zeros((100, 100))
        mock_get_embedding.return_value = np.zeros(512)
        mock_is_live.return_value = (True, 0.9)

        # Test data
        data = {
            'username': 'newstudent',
            'password1': 'testpass123',
            'password2': 'testpass123',
            'email': 'new@student.com',
            'face_data': 'base64_encoded_image_data',
        }

        response = self.client.post(reverse('student_register'), data)
        self.assertEqual(response.status_code, 302)  # Redirect after success
        self.assertTrue(
            self.User.objects.filter(username='newstudent').exists()
        )

    def test_invalid_form_submission(self):
        data = {
            'username': 'newstudent',
            # Missing required fields
        }
        response = self.client.post(reverse('student_register'), data)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(
            self.User.objects.filter(username='newstudent').exists()
        )

class TestTeacherDashboard(TestSetup):
    def setUp(self):
        super().setUp()
        self.unit = Unit.objects.create(
            name='Test Unit',
            code='TEST101',
            teacher=self.teacher_user
        )

    def test_teacher_dashboard_access(self):
        self.client.login(username='teacher', password='testpass123')
        response = self.client.get(reverse('teacher_dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'attendance/teacher_dashboard.html')

    def test_unauthorized_access(self):
        self.client.login(username='student', password='testpass123')
        response = self.client.get(reverse('teacher_dashboard'))
        self.assertEqual(response.status_code, 403)

class TestAdminDashboard(TestSetup):
    def test_admin_dashboard_access(self):
        self.client.login(username='admin', password='testpass123')
        response = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/admin_dashboard.html')

    def test_unauthorized_access(self):
        self.client.login(username='teacher', password='testpass123')
        response = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(response.status_code, 403)

class TestUserManagement(TestSetup):
    def test_edit_user(self):
        self.client.login(username='admin', password='testpass123')
        data = {
            'username': 'updated_username',
            'email': 'updated@test.com',
            'role': 'student'
        }
        response = self.client.post(
            reverse('edit_user', kwargs={'user_id': self.student_user.id}),
            data
        )
        self.assertEqual(response.status_code, 302)
        updated_user = self.User.objects.get(id=self.student_user.id)
        self.assertEqual(updated_user.username, 'updated_username')

    def test_delete_user(self):
        self.client.login(username='admin', password='testpass123')
        response = self.client.post(
            reverse('delete_user', kwargs={'user_id': self.student_user.id})
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(
            self.User.objects.filter(id=self.student_user.id).exists()
        )

class TestUpdateFaceData(TestSetup):
    @patch('face_recognition.detection.detect_face')
    @patch('face_recognition.recognition.get_face_embedding')
    def test_update_face_data(self, mock_get_embedding, mock_detect_face):
        self.client.login(username='student', password='testpass123')

        # Mock face recognition functions
        mock_detect_face.return_value = np.zeros((100, 100))
        mock_get_embedding.return_value = np.zeros(512)

        data = {
            'face_data': 'base64_encoded_image_data'
        }

        response = self.client.post(reverse('update_face_data'), data)
        self.assertEqual(response.status_code, 302)

        # Verify the student's face data was updated
        student = Student.objects.get(id=self.student.id)
        self.assertIsNotNone(student.face_image)
        self.assertIsNotNone(student.embedding_path)

def tearDown(self):
    # Clean up any files created during tests
    if os.path.exists('media/faces'):
        for file in os.listdir('media/faces'):
            os.remove(os.path.join('media/faces', file))
    if os.path.exists('media/embeddings'):
        for file in os.listdir('media/embeddings'):
            os.remove(os.path.join('media/embeddings', file))