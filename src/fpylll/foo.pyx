# cython: c_string_type=unicode, c_string_encoding=utf8
from .foo cimport greet

def greet_py(name):
    return greet(name)
