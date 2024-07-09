# -*- coding: utf-8 -*-

# pyorimap/quaternions_numba_gpu.py

"""
Quaternion operations accelerated using numba on GPU.

This module contains the following functions:

- `q4_mult(qa, qb)` - compute quaternion multiplication
- `q4_inv(qarr)` - compute the inverse of unit quaternions
- `q4_cosang2(qa, qb)` - cosine of the half angle between `qa` and `qb`
- `q4_disori_angle(qa, qb, qsym)` - compute the disorientation angle between `qa` and  `qb` (taking symmetries into account)
"""

import logging
import numpy as np
import cupy as cp
import quaternions_np as q4np

from numba import cuda, int32, float32

DTYPEf = np.float32
DTYPEi = np.int32
_EPS = 1e-9

@cuda.jit((float32[:], float32[:], float32[:]), fastmath=True, device=True)
def _device_q4_mult(qa, qb, qc):
    qc[0] = qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3]
    qc[1] = qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2]
    qc[2] = qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1]
    qc[3] = qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0]

@cuda.jit((float32[:], float32[:]), fastmath=True, device=True)
def _device_q4_inv(qa, qb):
    qb[0] =  qa[0]
    qb[1] = -qa[1]
    qb[2] = -qa[2]
    qb[3] = -qa[3]

@cuda.jit(float32(float32[:], float32[:]), fastmath=True, device=True)
def _device_q4_cosang2(qa, qb):
    return( min(abs(qa[0]*qb[0] + qa[1]*qb[1] + qa[2]*qb[2] + qa[3]*qb[3]), 1.) )

@cuda.jit((float32[:,:], float32[:,:], float32[:,:]))
def _kernel_q4_mult(qa, qb, qc):
    i = cuda.grid(1)
    if i < qa.shape[0]:
        _device_q4_mult(qa[i,:], qb[i,:], qc[i,:])

@cuda.jit((float32[:,:], float32[:,:]))
def _kernel_q4_inv(qa, qb):
    i = cuda.grid(1)
    if i < qa.shape[0]:
        _device_q4_inv(qa[i,:], qb[i,:])

@cuda.jit((float32[:,:], float32[:,:], float32[:,:], float32[:]), fastmath=True)
def _kernel_q4_disori_angle(qa, qb, qsym, ang):
    i = cuda.grid(1)
    if i < qa.shape[0]:
        # initialize arrays:
        b0 = 0.
        qa_inv = cuda.local.array(shape=4, dtype=float32)
        qc = cuda.local.array(shape=4, dtype=float32)

        # disorientation qc expressed in the frame of crystal a:
        _device_q4_inv(qa[i,:], qa_inv)
        _device_q4_mult(qa_inv, qb[i,:], qc)

        for j in range(len(qsym)):
            b1 = _device_q4_cosang2(qc, qsym[j,:])
            b0= np.maximum(b0, b1)

        ang[i] = np.arccos(b0) * 2 * 180/np.pi

def q4_mult(qa, qb, qc, nthreads=256):
    """
    Wrapper calling the kernel to compute quaternion multiplication
    `qc`=`qa`*`qb`.

    Numba GPU version, all input arrays already on GPU memory.

    Parameters
    ----------
    qa : ndarray
        array of quaternions or single quaternion on GPU memory.
    qb : ndarray
        array of quaternions or single quaternion on GPU memory.
    qc : ndarray
        array of quaternions or single quaternion on GPU memory.

    Examples
    --------
    >>> qa = q4np.q4_random(1024)
    >>> qb = qa[::-1,:]
    >>> qc = q4np.q4_mult(qa, qb)
    >>> qa_gpu = cp.asarray(qa, dtype=DTYPEf)
    >>> qb_gpu = cp.asarray(qb, dtype=DTYPEf)
    >>> qc_gpu = cp.zeros_like(qa_gpu)
    >>> q4_mult(qa_gpu, qb_gpu, qc_gpu, nthreads=128)
    >>> np.allclose(qc, cp.asnumpy(qc_gpu), atol=1e-6)
    True
    """
    threadsperblock = nthreads
    blockspergrid = (qa.shape[0] + (threadsperblock - 1)) // threadsperblock
    _kernel_q4_mult[blockspergrid, threadsperblock](qa, qb, qc)

def q4_inv(qa, qb, nthreads=256):
    """
    Wrapper calling the kernel to compute the inverse of unit quaternions.
    `qb`=inv(`qa`).

    Numba GPU version, all input arrays already on GPU memory.

    Parameters
    ----------
    qa : ndarray
        array of quaternions or single quaternion on GPU memory.
    qb : ndarray
        array of quaternions or single quaternion on GPU memory.

    Examples
    --------
    >>> qa = q4np.q4_random(1024)
    >>> qb = q4np.q4_inv(qa)
    >>> qa_gpu = cp.asarray(qa, dtype=DTYPEf)
    >>> qb_gpu = cp.zeros_like(qa_gpu)
    >>> q4_inv(qa_gpu, qb_gpu, nthreads=128)
    >>> np.allclose(qb, cp.asnumpy(qb_gpu), atol=1e-6)
    True
    """
    threadsperblock = nthreads
    blockspergrid = (qa.shape[0] + (threadsperblock - 1)) // threadsperblock
    _kernel_q4_inv[blockspergrid, threadsperblock](qa, qb)

def q4_disori_angle(qa, qb, qsym, ang, nthreads=256):
    """
    Wrapper calling the kernel to compute disorientation angle between
    `qa` and `qb` accounting for `qsym` symmetries.

    Numba GPU version, all input arrays already on GPU memory.

    Parameters
    ----------
    qa : ndarray
        array of quaternions or single quaternion on GPU memory.
    qb : ndarray
        array of quaternions or single quaternion on GPU memory.
    qsym : ndarray
        quaternion array of symmetry operations on GPU memory.
    ang : ndarray
        the minumum angle (in degrees) between quaternions `qa` and `qb`,
        taking symmetries into account, modified in place on GPU memory.

    Examples
    --------
    >>> qa = q4np.q4_random(1024)
    >>> qb = qa[::-1,:]
    >>> qsym = q4np.q4_sym_cubic()
    >>> ang = q4np.q4_disori_angle(qa, qb, qsym)
    >>> qa_gpu = cp.asarray(qa, dtype=DTYPEf)
    >>> qb_gpu = cp.asarray(qb, dtype=DTYPEf)
    >>> qsym_gpu = cp.asarray(qsym, dtype=DTYPEf)
    >>> ang_gpu = cp.zeros(qa_gpu.shape[0], dtype=DTYPEf)
    >>> q4_disori_angle(qa_gpu, qb_gpu, qsym_gpu, ang_gpu, nthreads=128)
    >>> np.allclose(ang, cp.asnumpy(ang_gpu), atol=1e-2)
    True
    """
    threadsperblock = nthreads
    blockspergrid = (qa.shape[0] + (threadsperblock - 1)) // threadsperblock
    _kernel_q4_disori_angle[blockspergrid, threadsperblock](qa, qb, qsym, ang)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

