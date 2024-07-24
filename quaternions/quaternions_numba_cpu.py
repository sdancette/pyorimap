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

from numba import njit, prange, int32, float32, boolean
from numba.types import Tuple

DTYPEf = np.float32
DTYPEi = np.int32
_EPS = 1e-7

@njit(float32[:,:](float32[:,:], float32), fastmath=True, parallel=True)
def q4_positive(qarr, _EPS=1e-7):
    """
    Return positive quaternion.

    If the first non-zero term of each quaternion is negative,
    the opposite quaternion is returned, quaternion q and -q
    representing the same rotation.

    Parameters
    ----------
    qarr : array_like
        (n, 4) array of quaternions or single quaternion of shape (4,).
    _EPS : float
        tolerance for non-zero values.

    Returns
    -------
    qpos : ndarray
        array of positive quaternions.

    Examples
    --------
    >>> qarr = np.array([[-1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,-1]]).astype(np.float32)
    >>> qpos = q4_positive(qarr, _EPS)
    >>> np.allclose(qpos, -qarr, atol=1e-6)
    True
    """
    qpos = qarr.copy()
    n = qpos.shape[0]

    for i in prange(n):
        opposite = False
        if (qpos[i,0] < -_EPS):
            opposite = True
        elif (qpos[i,0] >= -_EPS)*(qpos[i,0] < _EPS):
            if (qpos[i,1] < -_EPS):
                opposite = True
            elif (qpos[i,1] >= -_EPS)*(qpos[i,1] < _EPS):
                if (qpos[i,2] < -_EPS):
                    opposite = True
                elif (qpos[i,2] >= -_EPS)*(qpos[i,2] < _EPS):
                    if (qpos[i,3] < -_EPS):
                        opposite = True
        if opposite:
            qpos[i,:] *= -1.

    return qpos


@njit(float32[:,:](float32[:,:], float32[:,:]), fastmath=True, parallel=True)
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

@njit(float32[:](float32[:,:], float32[:,:]), fastmath=True, parallel=True)
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

@njit(Tuple((float32[:], int32[:]))(float32[:,:], float32[:,:], float32[:,:], int32), fastmath=True, parallel=True)
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
        (n, 4) array of quaternions of symmetry operations.
    method : int, default=1
        the method to compute disorientation: 1 or 2.
        Method 1 is faster.

    Returns
    -------
    ang : ndarray
        the minumum angle (in degrees) between quaternions `qa` and `qb`, taking symmetries into account.
    ii : ndarray
        the index of the i_th equivalent quaternion corresponding to the minimum disorientation.

    Examples
    --------
    >>> qa = q4np.q4_random(1)
    >>> qsym = q4np.q4_sym_cubic()
    >>> qequ = q4np.q4_mult(qa, qsym)
    >>> ang, ii = q4_disori_angle(qequ, qequ[::-1,:], qsym, method=1)
    >>> np.allclose(ang, np.zeros(24, dtype=np.float32), atol=0.1)
    True
    >>> qa = q4np.q4_random(1024)
    >>> qb = q4np.q4_random(1024)
    >>> ang0 = q4np.q4_disori_angle(qa, qb, qsym, method=1)
    >>> ang1, ii1 = q4_disori_angle(qa, qb, qsym, method=1)
    >>> ang2, ii2 = q4_disori_angle(qa, qb, qsym, method=2)
    >>> np.allclose(ang0, ang1, atol=0.1)
    True
    >>> np.allclose(ang1, ang2, atol=0.1)
    True
    >>> np.allclose(ii1, ii2)
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
    ii = np.zeros(n, dtype=np.int32)

    if method == 1:
        # disorientation expressed in the frame of crystal a:
        qc = q4_mult(q4_inv(qa), qb)
        for i in range(nsym):
            ang1 = q4_cosang2(qc, qsym[i:i+1,:]) # cosine of the half angle
            #ang = np.maximum(ang, ang1)
            for j in prange(n):
                #ang[j] = max(ang[j], ang1[j])
                if ang1[j] > ang[j]:
                    ang[j] = ang1[j]
                    ii[j] = i
    else:
        for i in range(nsym):
            qc = q4_mult(qa, qsym[i:i+1,:])
            ang1 = q4_cosang2(qc, qb) # cosine of the half angle
            #ang = np.maximum(ang, ang1)
            for j in prange(n):
                #ang[j] = max(ang[j], ang1[j])
                if ang1[j] > ang[j]:
                    ang[j] = ang1[j]
                    ii[j] = i

    # back to angles in degrees:
    for j in prange(n):
        ang[j] = 2*np.arccos(ang[j])*rad2deg
    return ang, ii

@njit(Tuple((float32[:,:], int32[:]))(float32[:,:], float32[:,:], float32[:,:], int32, int32), fastmath=True, parallel=True)
def q4_disori_quat(qa, qb, qsym, frame=0, method=1):
    """
    Disorientation quaternion between `qa` and `qb`, taking `qsym` symmetries into account.

    Parameters
    ----------
    qa : ndarray
        (n, 4) array of quaternions.
    qb : ndarray
        (n, 4) array of quaternions.
    qsym : ndarray
        (n, 4) quaternion array of symmetry operations.
    frame : int, default=0
        the frame to express the disorientation quaternion, 0='ref' or 1='crys_a'.
    method : int, default=1
        the method to compute disorientation: 1 or 2.
        Method 1 is faster.

    Returns
    -------
    qdis : ndarray
        quaternion representing the minimum disorientation from `qa` to `qb`, taking `qsym` symmetries into account.
    ii : ndarray
        the index of the i_th equivalent quaternion corresponding to the minimum disorientation.

    Examples
    --------
    >>> qa = np.array([[1,0,0,0]], dtype=np.float32)
    >>> qsym = q4np.q4_sym_cubic(); isym = 5
    >>> qb = q4_mult(qa, qsym[isym:isym+1])
    >>> qdis_ref, ii = q4_disori_quat(qa, qb, qsym, frame=0, method=1)
    >>> qdis_cra, ii = q4_disori_quat(qa, qb, qsym, frame=1, method=1)
    >>> np.allclose(qdis_ref, qdis_cra, atol=1e-6)
    True
    >>> qa = q4np.q4_random(1024)
    >>> qb = q4np.q4_random(1024)
    >>> ang, ii = q4_disori_angle(qa, qb, qsym, method=1)
    >>> qdis1, ii1 = q4_disori_quat(qa, qb, qsym, frame=0, method=1)
    >>> qdis2, ii2 = q4_disori_quat(qa, qb, qsym, frame=0, method=2)
    >>> np.allclose(qdis1, qdis2, atol=1e-6)
    True
    >>> ang1 = np.arccos(np.abs(qdis1[:,0]))*2*180/np.pi
    >>> ang2 = np.arccos(np.abs(qdis2[:,0]))*2*180/np.pi
    >>> np.allclose(ang, ang1, atol=0.1)
    True
    >>> np.allclose(ang, ang2, atol=0.1)
    True
    >>> qdis1, ii1 = q4_disori_quat(qa, qb, qsym, frame=1, method=1)
    >>> qdis2, ii2 = q4_disori_quat(qa, qb, qsym, frame=1, method=2)
    >>> np.allclose(qdis1, qdis2, atol=1e-6)
    True
    """
    if method == 1:
        ang, ii = q4_disori_angle(qa, qb, qsym, method=1)
        qa_inv = q4_inv(q4_mult(qa, qsym[ii]))
        if frame == 0:
            qdis = q4_mult(qb, qa_inv)
        else:
            qdis = q4_mult(qa_inv, qb)
    else:
        if qb.shape[0] == 1:
            qdis = np.zeros(qa.shape, dtype=np.float32)
        else:
            qdis = np.zeros(qb.shape, dtype=np.float32)

        n = qdis.shape[0]
        ii = np.zeros(n, dtype=np.int32)

        for isym, q in enumerate(qsym):
            qa_inv = q4_inv(q4_mult(qa, qsym[isym:isym+1]))
            if frame == 0:
                qtmp = q4_mult(qb, qa_inv)
            else:
                qtmp = q4_mult(qa_inv, qb)

            for j in prange(n):
                a0 = min(abs(qdis[j,0]), 1.)
                a1 = min(abs(qtmp[j,0]), 1.)
                if a1 > a0:
                    qdis[j,:] = qtmp[j,:]
                    ii[j] = isym

    qdis = q4_positive(qdis, _EPS)

    return qdis, ii

@njit(Tuple((float32[:], float32[:], float32[:], float32[:]))(float32[:,:], float32[:,:]), fastmath=True, parallel=True)
def q4_mean_disori(qarr, qsym):
    """
    Average orientation and disorientation (GOS and GROD).

    Parameters
    ----------
    qarr : ndarray
        (n, 4) array of quaternions.
    qsym : ndarray
        quaternion array of symmetry operations.

    Returns
    -------
    qavg : ndarray
        quaternion representing the average orientation of `qarr`.
    GROD : ndarray
        (n,) array of grain reference orientation deviation in degrees.
    GROD_stat : ndarray
        [mean, std, min, Q1, median, Q3, max] of the grain reference orientation deviation (GROD).
        GROD_stat[0] is the grain orientation spread (GOS), i.e. the average disorientation angle in degrees.
    theta_iter : ndarray
        convergence angle (degree) during the iterations for `qavg`.

    Examples
    --------
    >>> qa = np.atleast_2d(q4np.q4_random(1))
    >>> qarr = q4_mult(qa, q4np.q4_orispread(ncrys=1024, thetamax=2., misori=True))
    >>> qsym = q4np.q4_sym_cubic()
    >>> qavg, GROD, GROD_stat, theta_iter = q4_mean_disori(qarr, qsym)
    >>> np.allclose(qa[0], qavg, atol=1e-3)
    True
    >>> qavg2, GROD2, GROD_stat2, theta_iter2 = q4np.q4_mean_disori(qarr, qsym)
    >>> np.allclose(qavg, qavg2, atol=1e-3)
    True
    >>> np.allclose(GROD, GROD2, atol=0.5)
    True
    """

    ii = 0
    theta= 999.
    nitermax = 10
    theta_iter = np.zeros(nitermax, dtype=DTYPEf) - 1.
    ncrys = qarr.shape[0]
    while (theta > 0.1) and (ii < nitermax):
        if ii == 0:
            # initialize avg orientation:
            qref = qarr[0:1,:]

        # disorientation of each crystal wrt average orientation:
        qdis, _ = q4_disori_quat(qref, qarr, qsym, frame=1, method=1)

        qtmp = np.zeros((1,4), dtype=DTYPEf)
        for j in prange(ncrys):
            qtmp[0,0] += qdis[j,0]
            qtmp[0,1] += qdis[j,1]
            qtmp[0,2] += qdis[j,2]
            qtmp[0,3] += qdis[j,3]
        qtmp /= np.sqrt(qtmp[0,0]**2 + qtmp[0,1]**2 + qtmp[0,2]**2 + qtmp[0,3]**2)

        # q_mean=q_ref*q_sum/|q_sum|
        qavg = q4_mult(qref, qtmp)

        #### theta for convergence of qavg:
        theta = np.arccos(q4_cosang2(qref, qavg)[0])*2*180/np.pi
        theta_iter[ii] = theta
        #theta_iter[ii] = 0.
        qref = qavg

        ii += 1

    # angles:
    GROD = np.zeros(ncrys, dtype=DTYPEf)
    for j in prange(ncrys):
        GROD[j] = np.arccos(min(abs(qdis[j,0]), 1.))*2*180/np.pi

    GOS = GROD.mean()
    GROD_stat = np.array([GOS,
                 GROD.std(),
                 GROD.min(),
                 np.quantile(GROD, 0.25),
                 np.median(GROD),
                 np.quantile(GROD, 0.75),
                 GROD.max()], dtype=DTYPEf)
    qavg = qavg[0,:]
    theta_iter = theta_iter.astype(DTYPEf)
    theta_iter = theta_iter[theta_iter >= 0.]

    return qavg, GROD, GROD_stat, theta_iter


if __name__ == "__main__":
    import doctest
    doctest.testmod()

