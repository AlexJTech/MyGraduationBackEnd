# Generated by Django 4.2.11 on 2024-05-11 20:42

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('oAuth', '0004_books_is_delete'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Books',
        ),
    ]
