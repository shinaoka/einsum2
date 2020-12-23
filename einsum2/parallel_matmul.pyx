cimport numpy as np
import numpy as np
import cython
from cython.parallel import prange, parallel

ctypedef fused T1:
    double
    double complex

ctypedef fused T2:
    double
    double complex

def _par_matmul(A, B):
    assert A.dtype in [np.double, np.complex]
    assert B.dtype in [np.double, np.complex]

    dtype = np.result_type(A, B)
    if dtype == np.double:
        return _par_matmul_double(A, B)
    else:
        return _par_matmul_complex(A, B)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _par_matmul_double(double[:,:,:] A, double[:,:,:] B):
    if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
        raise ValueError("Invalid dimensions for matmul")

    cdef double[:,:,:] C = np.zeros((A.shape[0], A.shape[1], B.shape[2]))

    cdef int I,J,K,L,JL,IJL
    I,J,K,L = A.shape[0], A.shape[1], A.shape[2], B.shape[2]
    JL = J*L
    IJL = I*JL

    cdef int i,j,k,l,jl,ijl
    for ijl in prange(IJL, schedule='guided', nogil=True):
        i = ijl // JL
        jl = ijl % JL
        j = jl // L
        l = jl % L
        for k in range(K):
            C[i,j,l] = C[i,j,l] + A[i,j,k] * B[i,k,l]

    return C

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _par_matmul_complex(T1[:,:,:] A, T2[:,:,:] B):
    if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
        raise ValueError("Invalid dimensions for matmul")

    cdef double complex[:,:,:] C = np.zeros((A.shape[0], A.shape[1], B.shape[2]), dtype=np.complex)

    cdef int I,J,K,L,JL,IJL
    I,J,K,L = A.shape[0], A.shape[1], A.shape[2], B.shape[2]
    JL = J*L
    IJL = I*JL

    cdef int i,j,k,l,jl,ijl
    for ijl in prange(IJL, schedule='guided', nogil=True):
        i = ijl // JL
        jl = ijl % JL
        j = jl // L
        l = jl % L
        for k in range(K):
            C[i,j,l] = C[i,j,l] + A[i,j,k] * B[i,k,l]

    return C
