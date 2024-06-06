from setuptools import setup, Extension
import pybind11 
import os

current_path = os.path.abspath(os.path.dirname(__file__))
include_path = ['include']
absoulte_include_path = [os.path.join(current_path, path) for path in include_path]
another_include_path = [pybind11.get_include()]
absoulte_include_path.extend(another_include_path)

module = Extension('numpy_helper', sources=['numpy_helper.cpp'], extra_compile_args=['-O3', '-fopenmp'], include_dirs=absoulte_include_path, extra_link_args=['-fopenmp'])

setup(
    name='numpy_helper',
    version='1.0',
    ext_modules=[module],
)