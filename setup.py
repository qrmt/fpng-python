from setuptools import setup, Extension
import platform
import numpy

# Compiler flags
extra_compile_args = ["-fno-strict-aliasing"]
if platform.processor() == "x86_64":
    extra_compile_args += ["-msse4.1", "-mpclmul"]

# Define the extension module
ext_modules = [
    Extension(
        "pyfpng",
        sources=["src/fpng-python.cpp", "fpng/src/fpng.cpp"],
        include_dirs=["fpng/src/", numpy.get_include()],
        extra_compile_args=extra_compile_args,
    )
]

# Setup function
setup(
    ext_modules=ext_modules,
)
