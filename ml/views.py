from django.shortcuts import render,redirect
from django.http import HttpResponse
from .forms import UsageForm
from django.core.mail import send_mail
from auto_ml.settings import EMAIL_HOST_USER
from django.core.mail import EmailMessage
from django.contrib import messages
from .mlcode import *
import pandas as pd
import os
from django.contrib import messages


def home(request):
    return render(request,'ml/home.html')

def instructions(request):
    return render(request,'ml/instructions.html')

def example(request):
    return render(request,'ml/example.html')

def completed(request):
    return render(request,'ml/completed.html')

def usage(request):
    if request.method == 'POST':
        form=UsageForm(request.POST,request.FILES)
        if form.is_valid():
            email_id=form.cleaned_data.get('Email_Id')
            files_pret=request.FILES["Training_Data"]
            files_prep=request.FILES["Prediction_Data"]
            subject = 'Processed Data'
            if form['Date_Column']:
                target_variable=form.cleaned_data.get('Target_Variable')
                date_column=form.cleaned_data.get('Date_Column')
                message = 'The Target Variable is {0} and the date columns is {1}. Please find the corresponding predictions and feature importance graph attached to this mail.'.format('target_variable','date_column')






            else:
                target_variable=form.cleaned_data.get('Target_Variable')
                message = 'The Target Variable is {0} and there is no date column. Please find the corresponding predictions and feature importance graph attached to this mail.'.format('target_variable')
                date_column=None
            try:
                df_train=pd.read_csv(files_pret)
            except:
                df_train=pd.read_excel(files_pret)



            try:
                df_test=pd.read_csv(files_prep)

            except:
                df_train=pd.read_excel(files_prep)

            send_mail_final(target_variable=target_variable,df_train=df_train,df_test=df_test,email_id=email_id,message=message,date_column=date_column)
            return redirect('completed')

            ''' try:

                df_pred=auto_predictor(Target_Variable=target_variable,data_raw=df_train,n_valid=int(0.1*len(df_train)),data_to_predict=df_test,date_column=date_column)
                df_pred=pd.DataFrame(df_pred)
                df_pred.to_csv('predictions.csv')
                recepient = str(form['Email_Id'].value())
                email=EmailMessage(subject,message,EMAIL_HOST_USER,[recepient])
                email.attach_file('predictions.csv')
                email.attach_file('Feature Importance.png')
                xyz=1

            except:
                message='Something went wrong ! Please try again after checking all the fields.'
                recepient = str(form['Email_Id'].value())
                email=EmailMessage(subject,message,EMAIL_HOST_USER,[recepient])
                xyz=0





            #email.attach(files_send.name, files_send.read(), files_send.content_type)
            email.send()
            if xyz :
                os.remove('predictions.csv')
                os.remove('Feature Importance.png')




           # send_mail(subject,message, EMAIL_HOST_USER, [recepient], fail_silently = False)'''






    else :
        form=UsageForm()

    return render(request,'ml/index.html',{'form':form})
