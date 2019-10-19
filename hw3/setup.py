from distutils.core import setup, Extension

module = Extension("utils_cpp",
                   extra_compile_args=["-std=c++17"],
                   sources=["py_entry.cpp", "utils.cpp"])

setup(name="utils_cpp",
      version="1.0",
      description="Utilities of image processing. Writen by cpp",
      ext_modules=[module])
