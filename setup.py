""" Module setup """
from setuptools import setup, find_packages

setup(
    name='bff_positioning',
    version='0.1',
    description='A ML-based algorithm that enables accurate positioning from mmWave '
        'transmissions - with and without tracking',
    url='https://github.com/gante/mmWave-localization-learning.git',
    author='Joao Gante',
    author_email='joaofranciscocardosogante@gmail.com',
    packages=find_packages(),
    include_package_data=True
)
