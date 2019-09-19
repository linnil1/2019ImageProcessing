from distutils.core import setup, Extension

module = Extension('myimg',
                   extra_compile_args=['-std=c++17'],
                   sources = ['py_myimg.cpp'])

setup (name = 'MyImg',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module])
