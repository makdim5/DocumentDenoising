from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import Image, CSV


class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ('image',)


class CSVForm(forms.ModelForm):
    class Meta:
        model = CSV
        fields = ('eps', 'min_samples', 'file')


class RegisterUserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'password1', 'password2')
        widgets = {
            'username': forms.TextInput(),
            'password1': forms.PasswordInput(),
            'password2': forms.PasswordInput(),
        }
