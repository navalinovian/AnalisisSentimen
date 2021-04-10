# Generated by Django 3.1.7 on 2021-03-31 16:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_auto_20210331_1656'),
    ]

    operations = [
        migrations.RenameField(
            model_name='room',
            old_name='guest',
            new_name='guest_can_pause',
        ),
        migrations.AlterField(
            model_name='room',
            name='code',
            field=models.CharField(default='', max_length=8, unique=True),
        ),
    ]