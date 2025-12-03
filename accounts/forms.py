from django import forms
from .models import CustomUser, Student
from django.contrib.auth.forms import UserCreationForm
from django.core.exceptions import ValidationError
from django.utils import timezone

class StudentRegistrationForm(UserCreationForm):
    student_id = forms.CharField(max_length=20)
    face_data = forms.CharField(widget=forms.HiddenInput())  # base64 image

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2']

    def clean_student_id(self):
        student_id = self.cleaned_data.get('student_id')
        if Student.objects.filter(student_id=student_id).exists():
            raise ValidationError("A student with this ID already exists.")
        return student_id

    def clean_face_data(self):
        data = self.cleaned_data.get('face_data')
        if not data:
            raise ValidationError("Face data is required. Please capture your face before submitting.")
        return data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.role = 'student'
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            Student.objects.create(
                user=user,
                student_id=self.cleaned_data['student_id']
            )
        return user

class TeacherRegistrationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.role = 'teacher'
        if commit:
            user.save()
        return user

class AdminRegistrationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.role = 'admin'
        user.is_staff = True        # ✅ Needed to access /admin/
        user.is_superuser = True    # ✅ Optional but gives full access
        if commit:
            user.save()
        return user

class AdminEditUserForm(forms.ModelForm):
    """Form for administrators to edit user details"""

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'role', 'is_active']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Bootstrap classes and custom attributes
        for field in self.fields:
            self.fields[field].widget.attrs.update({
                'class': 'form-control',
                'placeholder': field.replace('_', ' ').title()
            })

        # Make role field as select
        self.fields['role'].widget = forms.Select(choices=[
            ('student', 'Student'),
            ('teacher', 'Teacher'),
            ('admin', 'Administrator')
        ])

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if CustomUser.objects.exclude(pk=self.instance.pk).filter(email=email).exists():
            raise forms.ValidationError("This email is already in use.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        # Add any additional validation logic here
        return cleaned_data
class StudentRegistrationForm(UserCreationForm):
    student_id = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter Student ID'
        })
    )
    face_data = forms.CharField(
        widget=forms.HiddenInput(),
        required=True
    )

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter Username'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter Email'
            })
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in ['password1', 'password2']:
            self.fields[field].widget.attrs.update({
                'class': 'form-control',
                'placeholder': 'Enter Password'
            })

    def clean_student_id(self):
        student_id = self.cleaned_data.get('student_id')
        if Student.objects.filter(student_id=student_id).exists():
            raise ValidationError("A student with this ID already exists.")
        return student_id

    def clean_face_data(self):
        data = self.cleaned_data.get('face_data')
        if not data:
            raise ValidationError("Face data is required. Please capture your face before submitting.")
        return data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.role = 'student'
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            Student.objects.create(
                user=user,
                student_id=self.cleaned_data['student_id']
            )
        return user

class TeacherRegistrationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter Username'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter Email'
            })
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in ['password1', 'password2']:
            self.fields[field].widget.attrs.update({
                'class': 'form-control',
                'placeholder': 'Enter Password'
            })

    def save(self, commit=True):
        user = super().save(commit=False)
        user.role = 'teacher'
        if commit:
            user.save()
        return user

class AdminRegistrationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter Username'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter Email'
            })
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in ['password1', 'password2']:
            self.fields[field].widget.attrs.update({
                'class': 'form-control',
                'placeholder': 'Enter Password'
            })

    def save(self, commit=True):
        user = super().save(commit=False)
        user.role = 'admin'
        user.is_staff = True
        user.is_superuser = True
        if commit:
            user.save()
        return user

class AdminEditUserForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'role', 'is_active']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields:
            self.fields[field].widget.attrs.update({
                'class': 'form-control',
                'placeholder': field.replace('_', ' ').title()
            })

        self.fields['role'].widget = forms.Select(
            choices=[
                ('student', 'Student'),
                ('teacher', 'Teacher'),
                ('admin', 'Administrator')
            ],
            attrs={'class': 'form-control'}
        )
        self.fields['is_active'].widget = forms.CheckboxInput(
            attrs={'class': 'form-check-input'}
        )

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if CustomUser.objects.exclude(pk=self.instance.pk).filter(email=email).exists():
            raise ValidationError("This email is already in use.")
        return email