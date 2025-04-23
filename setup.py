from setuptools import setup, Extension
import numpy

module = Extension(
    'rs_c',
    sources=['rs.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3'],
)

setup(
    name='rs_c',
    version='1.0',
    description='Recuit simulé en C',
    ext_modules=[module],
)
