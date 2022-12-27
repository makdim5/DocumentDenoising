from django.urls import path, include

from .views import MainView, AuthorizView, RegisterUser, logout_user

urlpatterns = [

    path("", AuthorizView.as_view(), name="ath"),
    path("main/", MainView.as_view(), name="main"),
    path('register/', RegisterUser.as_view(), name='register'),
    path('logout/', logout_user, name='logout'),
]
