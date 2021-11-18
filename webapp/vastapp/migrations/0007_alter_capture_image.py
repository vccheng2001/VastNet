# Generated by Django 3.2.9 on 2021-11-18 04:35

from django.db import migrations
import django_base64field.fields


class Migration(migrations.Migration):

    dependencies = [
        ('vastapp', '0006_alter_capture_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='capture',
            name='image',
            field=django_base64field.fields.Base64Field(blank=True, default='', max_length=900000, null=True),
        ),
    ]
