from django import forms
from django.core.validators import FileExtensionValidator


class UsageForm(forms.Form):
    Target_Variable=forms.CharField(max_length=100,help_text='Name of column to be predicted')
    Date_Column=forms.CharField(max_length=100,required=False,help_text='Name of column containing data if any(optional)')
    Email_Id = forms.EmailField(help_text='Your email id ')
    Training_Data=forms.FileField( validators=[FileExtensionValidator(allowed_extensions=['csv','xlsx'])],)
    Prediction_Data=forms.FileField( validators=[FileExtensionValidator(allowed_extensions=['csv','xlsx'])])

