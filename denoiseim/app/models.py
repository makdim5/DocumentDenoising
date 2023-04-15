from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models

from .validators import validate_file_extension


class Image(models.Model):
    image = models.ImageField(upload_to='images/%Y/%m/%d/')


class CSV(models.Model):
    eps = models.IntegerField(default=2,validators=[MinValueValidator(1), MaxValueValidator(100)])
    min_samples = models.IntegerField(default=4,validators=[MinValueValidator(1), MaxValueValidator(100)])
    file = models.FileField(upload_to='csv_files/%Y/%m/%d/', validators=[validate_file_extension])
