from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models

from .validators import validate_file_extension


class Image(models.Model):
    image = models.ImageField(upload_to='images/%Y/%m/%d/', verbose_name="Файл картинки")


class CSV(models.Model):
    eps = models.IntegerField(default=2,validators=[MinValueValidator(1), MaxValueValidator(100)], verbose_name="Минимальное расстояние между точками")
    min_samples = models.IntegerField(default=4,validators=[MinValueValidator(1), MaxValueValidator(100)], verbose_name="Минимальное число соседей в кластере")
    file = models.FileField(upload_to='csv_files/%Y/%m/%d/', verbose_name="Файл")
