# setup.py

from setuptools import setup, find_packages

setup(
    name='InceptionCRO',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'graphviz',
        'jupyter'
    ],
    author='David Pineda Peña',
    description='Optimización de Módulos Inception Dinámicos mediante CRO'
)
