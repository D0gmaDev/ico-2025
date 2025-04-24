from setuptools import setup, Extension
import numpy

rs_module = Extension(
    'rs_c',
    sources=['ico.c', 'rs.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3'],
)

tabu_module = Extension(
    'tabu',
    sources=['ico.c', 'tabu.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3'],
)

setup(
    name='optimization_algorithms',
    version='1.0',
    description='Recuit simulé et Tabu Search en C',
    ext_modules=[rs_module, tabu_module],
)
