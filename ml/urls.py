from django.urls import path
from . import views
urlpatterns = [
    path('', views.home,name='ml-home'),
    path('how-to-use/',views.instructions,name='instructions'),
    path('data-collection-form/',views.usage,name='use'),
    path('example/',views.example,name='example'),
    path('completed/',views.completed,name='completed')

]
