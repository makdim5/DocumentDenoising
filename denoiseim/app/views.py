import pandas as pd
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.views import LoginView
from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import CreateView
from sklearn.cluster import DBSCAN

from .forms import ImageForm, RegisterUserForm, CSVForm

from PIL import Image
from pytesseract import pytesseract
import requests


def get_text_from_img(path_to_image):
    path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.tesseract_cmd = path_to_tesseract

    img = Image.open(path_to_image)

    return pytesseract.image_to_string(img)


class AuthorizView(LoginView):
    form_class = AuthenticationForm
    template_name = "app/authorization.html"

    def get_context_data(self, *, object_list=None, **kwargs):
        return super().get_context_data(**kwargs, title="Авторизация")

    def get_success_url(self):
        return reverse_lazy('main')


class RegisterUser(CreateView):
    form_class = RegisterUserForm
    template_name = 'app/registration.html'
    success_url = reverse_lazy('ath')

    def get_context_data(self, *, object_list=None, **kwargs):
        return super().get_context_data(**kwargs, title="Регистрация")

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('ath')


def logout_user(request):
    logout(request)
    return redirect('ath')


class MainView(View):
    def get(self, request):

        return render(request, "app/index.html", context={"form": ImageForm()})

    def post(self, request):
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            res_filepath = "app/static/app/images/new.png"
            img_obj = form.instance

            with open(img_obj.image.url[1:], mode="rb") as file:
                try:
                    x = requests.post('http://localhost:80', data=file.read())

                    with open(res_filepath, "wb") as file:
                        file.write(x.content)
                except requests.exceptions.ConnectionError:
                    pass

            return render(request, 'app/index.html',
                          {'form': form, 'img_obj': img_obj, "text": get_text_from_img(res_filepath)})


class DBSCANView(View):
    def get(self, request):
        return render(request, "app/dbscan_template.html", context={"form": CSVForm()})

    def post(self, request):
        try:
            form = CSVForm(request.POST, request.FILES)
            if form.is_valid():
                form.save()

                file_obj = form.instance

                nodes_df = pd.read_csv(file_obj.file.url[1:])
                print(nodes_df)

                af = DBSCAN(eps=file_obj.eps, min_samples=file_obj.min_samples).fit(nodes_df)
                res = []
                nodes_df["cluster"] = af.labels_

                if max(af.labels_) != -1:
                    for i in range(max(af.labels_) + 1):
                        df = nodes_df[nodes_df["cluster"] == i]
                        res.append([
                            df["x"].to_list(),
                            df["y"].to_list(),
                            df["z"].to_list()
                        ])
                else:
                    res.append([
                        nodes_df["x"].to_list(),
                        nodes_df["y"].to_list(),
                        nodes_df["z"].to_list()
                    ])
                return render(request, 'app/dbscan_template.html',
                              {'form': form, 'file_obj': file_obj, "points": res})
        except:
            return render(request, "app/dbscan_template.html", context={"form": CSVForm()})
