from django import forms
from .models import Unit
from accounts.models import CustomUser

from django import forms
from .models import Unit

class UnitForm(forms.ModelForm):
    """Form for creating and editing units"""

    class Meta:
        model = Unit
        fields = ['code', 'name', 'description', 'teacher']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Bootstrap classes and placeholders
        for field in self.fields:
            self.fields[field].widget.attrs.update({
                'class': 'form-control',
                'placeholder': f'Enter unit {field}'
            })

    def clean_code(self):
        code = self.cleaned_data.get('code')
        if Unit.objects.filter(code=code).exclude(pk=self.instance.pk if self.instance else None).exists():
            raise forms.ValidationError("A unit with this code already exists.")
        return code