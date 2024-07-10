# -*- coding: utf-8 -*-

# pyorimap/quaternions_numba_cpu.py

"""
Quaternion operations accelerated using numba on CPU.

This module contains the following functions:

- `q4_mult(qa, qb)` - compute quaternion multiplication
- `q4_inv(qarr)` - compute the inverse of unit quaternions
- `q4_cosang2(qa, qb)` - cosine of the half angle between `qa` and `qb`
- `q4_disori_angle(qa, qb, qsym)` - compute the disorientation angle between `qa` and  `qb` (taking symmetries into account)
"""

import logging
import numpy as np
from pyorimap.quaternions import quaternions_np as q4np

from numba import njit, prange, int32, float32

DTYPEf = np.float32
DTYPEi = np.int32
_EPS = 1e-9

@njit(float32[:,:](float32[:,:],float32[:,:]), fastmath=True, parallel=True)
def q4_mult(qa, qb):
    """
    Quaternion multiplication using numba on CPU.

    Parameters
    ----------
    qa : ndarray
        (n, 4) array of quaternions.
    qb : ndarray
        (n, 4) array of quaternions.

    Returns
    -------
    qc : ndarray
        the result of the quaternion multiplication `qa`*`qb`.
        Type np.float32 by default.

    Examples
    --------
    >>> qa = q4np.q4_random(n=1024)
    >>> qb = q4np.q4_random(n=1024)
    >>> qc = q4np.q4_mult(qa, qb)
    >>> qc1 = q4_mult(qa, qb)
    >>> np.allclose(qc, qc1, atol=1e-6)
    True
    >>> qa = qa[0]
    >>> qb = q4np.q4_random(n=1024)
    >>> qc = q4np.q4_mult(qa, qb)
    >>> qc1 = q4_mult(np.atleast_2d(qa), qb)
    >>> np.allclose(qc, qc1, atol=1e-6)
    True
    >>> qa = q4np.q4_random(n=1024)
    >>> qb = qb[0]
    >>> qc = q4np.q4_mult(qa, qb)
    >>> qc1 = q4_mult(qa, np.atleast_2d(qb))
    >>> np.allclose(qc, qc1, atol=1e-6)
    True
    """

    na = qa.shape[0]
    nb = qb.shape[0]
    if na == nb:
        qc = np.zeros((na,4), dtype=np.float32)
        for i in prange(na):
            qc[i,0] = qa[i,0]*qb[i,0] - qa[i,1]*qb[i,1] - qa[i,2]*qb[i,2] - qa[i,3]*qb[i,3]
            qc[i,1] = qa[i,0]*qb[i,1] + qa[i,1]*qb[i,0] + qa[i,2]*qb[i,3] - qa[i,3]*qb[i,2]
            qc[i,2] = qa[i,0]*qb[i,2] - qa[i,1]*qb[i,3] + qa[i,2]*qb[i,0] + qa[i,3]*qb[i,1]
            qc[i,3] = qa[i,0]*qb[i,3] + qa[i,1]*qb[i,2] - qa[i,2]*qb[i,1] + qa[i,3]*qb[i,0]
    elif nb == 1:
        qc = np.zeros((na,4), dtype=np.float32)
        for i in prange(na):
            qc[i,0] = qa[i,0]*qb[0,0] - qa[i,1]*qb[0,1] - qa[i,2]*qb[0,2] - qa[i,3]*qb[0,3]
            qc[i,1] = qa[i,0]*qb[0,1] + qa[i,1]*qb[0,0] + qa[i,2]*qb[0,3] - qa[i,3]*qb[0,2]
            qc[i,2] = qa[i,0]*qb[0,2] - qa[i,1]*qb[0,3] + qa[i,2]*qb[0,0] + qa[i,3]*qb[0,1]
            qc[i,3] = qa[i,0]*qb[0,3] + qa[i,1]*qb[0,2] - qa[i,2]*qb[0,1] + qa[i,3]*qb[0,0]
    elif na == 1:
        qc = np.zeros((nb,4), dtype=np.float32)
        for i in prange(nb):
            qc[i,0] = qa[0,0]*qb[i,0] - qa[0,1]*qb[i,1] - qa[0,2]*qb[i,2] - qa[0,3]*qb[i,3]
            qc[i,1] = qa[0,0]*qb[i,1] + qa[0,1]*qb[i,0] + qa[0,2]*qb[i,3] - qa[0,3]*qb[i,2]
            qc[i,2] = qa[0,0]*qb[i,2] - qa[0,1]*qb[i,3] + qa[0,2]*qb[i,0] + qa[0,3]*qb[i,1]
            qc[i,3] = qa[0,0]*qb[i,3] + qa[0,1]*qb[i,2] - qa[0,2]*qb[i,1] + qa[0,3]*qb[i,0]

    return qc

@njit(float32[:,:](float32[:,:]), fastmath=True, parallel=False)
def q4_inv(qarr):
    """
    Inverse of unit quaternion based on its conjugate.

    Parameters
    ----------
    qarr : ndarray
        (n, 4) array of quaternions.

    Returns
    -------
    qinv : ndarray
        the conjugate of `qarr`,
        (n, 4) array of quaternions.

    Examples
    --------
    >>> qa = np.atleast_2d([0,1,0,0]).astype(np.float32)
    >>> qinv = q4_inv(qa)
    >>> np.allclose(qinv[0], np.array([0,-1,0,0], dtype=np.float32), atol=1e-6)
    True
    >>> qa = q4np.q4_random(n=1024)
    >>> qinv = q4_inv(qa)
    """
    n = qarr.shape[0]
    qinv = qarr.copy()
    for i in prange(n):
        qinv[i,1] *= -1
        qinv[i,2] *= -1
        qinv[i,3] *= -1
    return qinv

@njit(float32[:](float32[:,:],float32[:,:]), fastmath=True, parallel=True)
def q4_cosang2(qa, qb):
    """
    Returns the cosine of the half angle between quaternions `qa` and `qb`.

    Parameters
    ----------
    qa : ndarray
        (n, 4) array of quaternions.
    qb : ndarray
        (n, 4) array of quaternions.

    Returns
    -------
    ang : ndarray
        the cosine of the half angle between quaternions `qa` and `qb`.

    Examples
    --------
    >>> qa = q4np.q4_random(1024)
    >>> ang = np.random.rand(1024)*180.
    >>> qrot = q4np.q4_from_axis_angle(np.random.rand(1024,3), ang)
    >>> qb = q4_mult(qa, qrot)
    >>> cosa2 = q4_cosang2(qa, qb)
    >>> aback = np.degrees(2*np.arccos(cosa2))
    >>> aback1 = np.degrees(2*np.arccos( q4np.q4_cosang2(qa, qb) ))
    >>> np.allclose(aback, ang, atol=0.1)
    True
    >>> np.allclose(aback, aback1, atol=0.1)
    True
    """
    na = qa.shape[0]
    nb = qb.shape[0]
    rad2deg = 180./np.pi
    if na == nb:
        ang = np.zeros(na, dtype=np.float32)
        for i in prange(na):
            ang[i] = min(abs(qa[i,0]*qb[i,0] +
                             qa[i,1]*qb[i,1] +
                             qa[i,2]*qb[i,2] +
                             qa[i,3]*qb[i,3]), 1.)
    elif nb == 1:
        ang = np.zeros(na, dtype=np.float32)
        for i in prange(na):
            ang[i] = min(abs(qa[i,0]*qb[0,0] +
                             qa[i,1]*qb[0,1] +
                             qa[i,2]*qb[0,2] +
                             qa[i,3]*qb[0,3]), 1.)
    elif na == 1:
        ang = np.zeros(nb, dtype=np.float32)
        for i in prange(nb):
            ang[i] = min(abs(qa[0,0]*qb[i,0] +
                             qa[0,1]*qb[i,1] +
                             qa[0,2]*qb[i,2] +
                             qa[0,3]*qb[i,3]), 1.)
    return ang

@njit(float32[:](float32[:,:],float32[:,:],float32[:,:],int32), fastmath=True, parallel=True)
def q4_disori_angle(qa, qb, qsym, method=1):
    """
    Disorientation angle (degrees) between `qa` and `qb`, taking `qsym` symmetries into account.

    Parameters
    ----------
    qa : ndarray
        (n, 4) array of quaternions.
    qb : ndarray
        (n, 4) array of quaternions.
    qsym : ndarray
        (n, 4) array of quaternions.
    method : int, default=1
        the method to compute disorientation: 1 or 2.
        Method 1 is faster.

    Returns
    -------
    ang : ndarray
        the minumum angle (in degrees) between quaternions `qa` and `qb`, taking symmetries into account.

    Examples
    --------
    >>> qa = q4np.q4_random(1)
    >>> qsym = q4np.q4_sym_cubic()
    >>> qequ = q4np.q4_mult(qa, qsym)
    >>> ang = q4_disori_angle(qequ, qequ[::-1,:], qsym, method=1)
    >>> np.allclose(ang, np.zeros(24, dtype=np.float32), atol=0.1)
    True
    >>> qa = q4np.q4_random(1024)
    >>> qb = q4np.q4_random(1024)
    >>> ang1 = q4_disori_angle(qa, qb, qsym, method=1)
    >>> ang2 = q4_disori_angle(qa, qb, qsym, method=2)
    >>> np.allclose(ang1, ang2, atol=0.1)
    True
    """
    rad2deg = 180./np.pi
    na = qa.shape[0]
    nb = qb.shape[0]
    nsym = qsym.shape[0]
    if (na == nb) or (nb == 1):
        n = na
    elif na == 1:
        n = nb
    ang = np.zeros(n, dtype=np.float32)

    if method == 1:
        # disorientation expressed in the frame of crystal a:
        qc = q4_mult(q4_inv(qa), qb)
        for i in range(nsym):
            ang1 = q4_cosang2(qc, qsym[i:i+1,:]) # cosine of the half angle
            #ang = np.maximum(ang, ang1)
            for j in prange(n):
                ang[j] = max(ang[j], ang1[j])
    else:
        for i in range(nsym):
            qc = q4_mult(qa, qsym[i:i+1,:])
            ang1 = q4_cosang2(qc, qb) # cosine of the half angle
            #ang = np.maximum(ang, ang1)
            for j in prange(n):
                ang[j] = max(ang[j], ang1[j])

    # back to angles in degrees:
    for j in prange(n):
        ang[j] = 2*np.arccos(ang[j])*rad2deg
    return ang

if __name__ == "__main__":
    import doctest
    doctest.testmod()

