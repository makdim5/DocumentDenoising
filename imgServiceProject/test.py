import os

import requests

with open(r"C:\Users\makan\Desktop\doc_denoising\denoising_dirty_documents\test\test\40.png", mode="rb") as file:
    try:
        x = requests.post('http://localhost:80', data=file.read())

        with open("new2.png", "wb") as file:
            file.write(x.content)

        os.system("new2.png")

    except requests.exceptions.ConnectionError:
        print("Нет соединения или ошибка сервера!")
