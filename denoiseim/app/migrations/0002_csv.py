# Generated by Django 4.2 on 2023-04-15 19:40

import app.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='CSV',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.FileField(upload_to='csv_files/%Y/%m/%d/', validators=[app.validators.validate_file_extension])),
            ],
        ),
    ]