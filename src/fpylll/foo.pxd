from libcpp.string cimport string
cdef extern from "fplll/foo.h" namespace "fplll":
    cdef string greet(string name)
