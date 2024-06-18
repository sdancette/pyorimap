# -*- coding: utf-8 -*-

# pyorimap/quaternions_numba_cpu.py

"""
Quaternion operations accelerated using numba on CPU.

This module contains the following functions:

- `q4_mult(qa, qb)` - compute quaternion multiplication
"""

import logging
import numpy as np
import quaternions_np as q4np

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

@njit(float32[:](float32[:,:],float32[:,:]), fastmath=True, parallel=True)
def q4_angle(qa, qb):
    """
    Returns the angle (degrees) between quaternions `qa` and `qb`.

    Parameters
    ----------
    qa : ndarray
        (n, 4) array of quaternions.
    qb : ndarray
        (n, 4) array of quaternions.

    Returns
    -------
    ang : ndarray
        the angle (in degrees) between quaternions `qa` and `qb`.

    Examples
    --------
    >>> qa = q4np.q4_random(1024)
    >>> ang = np.random.rand(1024)*180.
    >>> qrot = q4np.q4_from_axis_angle(np.random.rand(1024,3), ang)
    >>> qb = q4_mult(qa, qrot)
    >>> aback = q4_angle(qa, qb)
    >>> aback1 = q4np.q4_angle(qa, qb)
    >>> np.allclose(aback, ang, atol=1e-1)
    True
    >>> np.allclose(aback, aback1, atol=1e-1)
    True
    """
    na = qa.shape[0]
    nb = qb.shape[0]
    rad2deg = 180./np.pi
    if na == nb:
        ang = np.zeros(na, dtype=np.float32)
        for i in prange(na):
            ang[i] = 2.*np.arccos(min(abs(qa[i,0]*qb[i,0] +
                                          qa[i,1]*qb[i,1] +
                                          qa[i,2]*qb[i,2] +
                                          qa[i,3]*qb[i,3]), 1.))*rad2deg
    elif nb == 1:
        ang = np.zeros(na, dtype=np.float32)
        for i in prange(na):
            ang[i] = 2.*np.arccos(min(abs(qa[i,0]*qb[0,0] +
                                          qa[i,1]*qb[0,1] +
                                          qa[i,2]*qb[0,2] +
                                          qa[i,3]*qb[0,3]), 1.))*rad2deg
    elif na == 1:
        ang = np.zeros(nb, dtype=np.float32)
        for i in prange(nb):
            ang[i] = 2.*np.arccos(min(abs(qa[0,0]*qb[i,0] +
                                          qa[0,1]*qb[i,1] +
                                          qa[0,2]*qb[i,2] +
                                          qa[0,3]*qb[i,3]), 1.))*rad2deg
    return ang

@njit(float32[:](float32[:,:],float32[:,:],float32[:,:]), fastmath=True, parallel=True)
def q4_disori_angle(qa, qb, qsym):
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

    Returns
    -------
    ang : ndarray
        the minumum angle (in degrees) between quaternions `qa` and `qb`, taking symmetries into account.

    Examples
    --------
    >>> qa = q4np.q4_random(1)
    >>> qsym = q4np.q4_sym_cubic()
    >>> qequ = q4np.q4_mult(qa, qsym)
    >>> ang = q4_disori_angle(qequ, qequ[::-1,:], qsym)
    >>> np.allclose(ang, np.zeros(24, dtype=np.float32))
    True
    """
    na = qa.shape[0]
    nb = qb.shape[0]
    nsym = qsym.shape[0]
    if (na == nb) or (nb == 1):
        n = na
    elif na == 1:
        n = nb
    ang = np.zeros(n, dtype=np.float32) + 99999

    for i in range(nsym):
        qc = q4_mult(qa, qsym[i:i+1,:])
        ang1 = q4_angle(qc, qb)
        #ang = np.minimum(ang, ang1)
        for j in prange(n):
            ang[j] = min(ang[j], ang1[j])
    return ang


