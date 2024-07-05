from .gmp.mpz cimport mpz_t
from .gmp.mpf cimport mpf_t
from .fplll.fplll cimport FloatType, Z_NR, PrunerMetric, IntType
from .fplll.fplll cimport BKZParam as BKZParam_c
from .fplll.fplll cimport PrunerMetric
import numpy
from numpy.__init__ cimport float32_t, float64_t

cdef object check_float_type(object float_type)
cdef object check_int_type(object int_type)
cdef int preprocess_indices(int &i, int &j, int m, int n) except -1
cdef int check_precision(int precision) except -1
cdef int check_eta(float eta) except -1
cdef int check_delta(float delta) except -1
cdef PrunerMetric check_pruner_metric(object metric)
cdef void npy_float32_scale_to_mpz(mpz_t mpz_val, float32_t value)
cdef void npy_float64_scale_to_mpz(mpz_t mpz_val, float64_t value)
