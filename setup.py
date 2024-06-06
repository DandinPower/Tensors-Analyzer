from setuptools import setup, find_packages

setup(
    name="tensors_analyzer",
    version="0.1",
    packages=find_packages(),
    description='A simple tool to analyze tensors',
    author='Joseph Liaw',
    author_email='tomhot246@gmail.com',
    url='https://github.com/DandinPower/Tensor-Analyzer',
    install_requires=[
        'numpy',
        'torch',
        'pybind11',
    ],
)