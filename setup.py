from setuptools import setup, Extension

import platform

# Setup extension compile arguments
extra_compile_args = ['-fno-strict-aliasing']
# If on x86_64, enable SSE
if platform.processor() == 'x86_64':
    extra_compile_args += ['-msse4.1', '-mpclmul']

setup(
    name='pyfpng',
    version='0.1',
    description='Python bindings for fpng',
    author='Oskar Vuola',
    packages=['pyfpng'],
    ext_modules=[Extension('pyfpng.libpyfpng', ['src/fpng_py.cpp', 'fpng/src/fpng.cpp'], include_dirs=['fpng/src/'], extra_compile_args=extra_compile_args)],
    install_requires=["numpy>=1.20.0"]
)
