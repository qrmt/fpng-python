from setuptools import setup, Extension

import platform
import numpy

# Setup extension compile arguments
extra_compile_args = ['-fno-strict-aliasing']
# If on x86_64, enable SSE
if platform.processor() == 'x86_64':
    extra_compile_args += ['-msse4.1', '-mpclmul']

setup(
    name='fpng-python',
    version='0.1',
    description='Python bindings for fpng',
    author='Oskar Vuola',
    ext_modules=[Extension('pyfpng', ['src/fpng-python.cpp', 'fpng/src/fpng.cpp'], include_dirs=['fpng/src/', numpy.get_include()], extra_compile_args=extra_compile_args)],
    install_requires=["numpy>=1.20.0"]
)
