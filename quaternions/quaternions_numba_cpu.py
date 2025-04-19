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
import scipy.ndimage as ndi
from pyorimap.quaternions import quaternions_np as q4np

from numba import njit, prange, int32, float32, boolean
from numba.types import Tuple

DTYPEf = np.float32
DTYPEi = np.int32
_EPS = 1e-7

qsymCubic = q4np.q4_sym_cubic()
qsymHex =   q4np.q4_sym_hex()
qsymTetra = q4np.q4_sym_tetra()
qsymOrtho = q4np.q4_sym_ortho()
qsymMono =  q4np.q4_sym_mono()

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

@njit(Tuple((float32[:,:], int32[:]))(float32[:,:], float32[:,:]), fastmath=True, parallel=True)
def q4_to_FZ(qarr, qsym):
    """
    Move quaternions to Fundamental Zone based on crystal symmetries.

    The Fundamental Zone corresponds to the i_th equivalent orientation with the lowest angle wrt the reference frame.

    Parameters
    ----------
    qarr : ndarray
        (n, 4) array of quaternions.
    qsym : ndarray
        quaternion array of symmetry operations.

    Returns
    -------
    qFZ : ndarray
        equivalent quaternion array in the Fundamental Zone.
    ii : ndarray
        if `return_index`=True, the index of the i_th equivalent quaternion corresponding to Fundamental Zone.

    Examples
    --------
    >>> qsym = q4np.q4_sym_cubic()
    >>> qFZ, isym = q4_to_FZ(qsym, qsym)
    >>> ref = np.tile(np.array([1,0,0,0], dtype=DTYPEf), 24).reshape(24, -1)
    >>> np.allclose(qFZ, ref, atol=1e-6)
    True
    >>> qarr = q4np.q4_random(1024)
    >>> qFZ, isym = q4_to_FZ(qarr, qsym)
    >>> qFZ2, isym2 = q4np.q4_to_FZ(qarr, qsym, return_index=True)
    >>> np.allclose(qFZ, qFZ2, atol=1e-6)
    True
    >>> np.allclose(isym, isym2)
    True
    """
    n = qarr.shape[0]
    qFZ = np.zeros((n, 4), dtype=np.float32)
    ii = np.zeros(n, dtype=np.int32)

    for isym, q in enumerate(qsym):
        qequ = q4_mult(qarr, qsym[isym:isym+1,:])
        for j in prange(n):
            a0 = min(abs(qFZ[j,0]), 1.)
            a1 = min(abs(qequ[j,0]), 1.)
            if a1 > a0:
                qFZ[j,:] = qequ[j,:]
                ii[j] = isym
    qFZ = q4_positive(qFZ, _EPS)

    return qFZ, ii

@njit(float32[:,:](float32[:,:]), fastmath=True, parallel=True)
def q4_from_eul(eul):
    """
    Converts Bunge Euler angles (in degrees) to quaternions.

    Parameters
    ----------
    eul : ndarray
        (ncrys, 3) array of Bunge Euler angles in degrees ['phi1', 'Phi', 'phi2'].

    Returns
    -------
    qarr : ndarray
        quaternion array of shape (ncrys, 4) and type np.float32.

    Notes
    -----
    Bunge Euler angles correspond to a "Z X Z" sequence of successive rotations from the sample frame to the crystal frame,
    where the 2nd and 3rd rotation apply on the rotated frame resulting from the previous rotations.
    The present implementation follows the conventions detailed in the documentation of the
    orilib routines by R. Quey at https://sourceforge.net/projects/orilib/.

    Examples
    --------
    >>> qarr = q4np.q4_random(n=1024)
    >>> eul = q4np.q4_to_eul(qarr)
    >>> qback = q4_from_eul(eul)
    >>> np.allclose(qarr, qback, atol=1e-6)
    True
    """
    ncrys = len(eul)

    qarr = np.zeros((ncrys,4), dtype=np.float32)

    for j in prange(ncrys):
        phi1 = eul[j,0] * np.pi/180.
        Phi  = eul[j,1] * np.pi/180.
        phi2 = eul[j,2] * np.pi/180.

        qarr[j,0] = np.cos(Phi/2)*np.cos((phi1+phi2)/2)
        qarr[j,1] = np.sin(Phi/2)*np.cos((phi1-phi2)/2)
        qarr[j,2] = np.sin(Phi/2)*np.sin((phi1-phi2)/2)
        qarr[j,3] = np.cos(Phi/2)*np.sin((phi1+phi2)/2)

    # positive quaternion:
    qarr = q4_positive(qarr, _EPS)

    return qarr

@njit(float32[:,:,:](float32[:,:]), fastmath=True, parallel=True)
def q4_to_mat(qarr):
    """
    Compute rotation matrix from crystal quaternion.

    Parameters
    ----------
    qarr : ndarray
        (n, 4) array of quaternions or single quaternion of shape (4,).

    Returns
    -------
    R : ndarray
        (n, 3, 3) array of rotation matrices or single rotation matrix
        of shape (3, 3) depending on input. Dtype is np.float32 by default.

    Notes
    -----
    When the quaternions represent rotations from the sample frame
    to the crystal frame (as obtained for example from Bunge Euler angles),
    the columns of the resulting rotation matrix correspond to the unit vectors
    of the sample (macroscopic) frame expressed in the crystal frame.
    The present implementation follows the conventions detailed in the documentation of the
    orilib routines by R. Quey at https://sourceforge.net/projects/orilib/.

    Examples
    --------
    >>> R = q4np.mat_from_eul([[0,45,30],[10,20,30],[0,180,10]])
    >>> qarr = q4np.q4_from_mat(R)
    >>> Rback = q4_to_mat(qarr)
    >>> np.allclose(R, Rback, atol=1e-6)
    True
    >>> qarr = q4np.q4_random(1024)
    >>> Ra = q4np.q4_to_mat(qarr)
    >>> Rb = q4_to_mat(qarr)
    >>> np.allclose(Ra, Rb, atol=1e-6)
    True
    """
    ncrys = len(qarr)
    R = np.zeros((ncrys, 3, 3), dtype=DTYPEf)

    for j in prange(ncrys):
        q0 = qarr[j,0]; q1 = qarr[j,1]; q2 = qarr[j,2]; q3 = qarr[j,3]
        R[j,0,0] = q0**2+q1**2-1./2
        R[j,1,0] = q1*q2-q0*q3
        R[j,2,0] = q1*q3+q0*q2
        R[j,0,1] = q1*q2+q0*q3
        R[j,1,1] = q0**2+q2**2-1./2
        R[j,2,1] = q2*q3-q0*q1
        R[j,0,2] = q1*q3-q0*q2
        R[j,1,2] = q2*q3+q0*q1
        R[j,2,2] = q0**2+q3**2-1./2
    R *= 2.

    return R

@njit(float32[:,:](float32[:,:,:], float32[:,:]), fastmath=True, parallel=True)
def matvec_mult(A, b):
    """
    Stacked matrix-vector multiplication using numba on CPU.

    Parameters
    ----------
    A : ndarray
        (ncrys, n, n) array of matrices.
    b : ndarray
        (ncrys, n) array of matrices.

    Returns
    -------
    C : ndarray
        (ncrys, n) array, result of the matrix multiplication `A`*`b`.

    Examples
    --------
    >>> A = np.random.rand(1024,3,3).astype(DTYPEf)
    >>> b = np.random.rand(3).astype(DTYPEf)
    >>> C1 = np.matvec(A,b)
    >>> C2 = matvec_mult(A, np.atleast_2d(b))
    >>> np.allclose(C1, C2, atol=1e-6)
    True
    >>> A = np.random.rand(1024,3,3).astype(DTYPEf)
    >>> b = np.random.rand(1024,3).astype(DTYPEf)
    >>> C1 = np.matvec(A,b)
    >>> C2 = matvec_mult(A,b)
    >>> np.allclose(C1, C2, atol=1e-6)
    True
    """

    ncrysA, mA, nA = A.shape
    ncrysB, nB = b.shape

    if (nA == nB) & (mA == nA):
        C = np.zeros((ncrysA,nB), dtype=DTYPEf)
        for icrys in prange(ncrysA):
            for i in range(mA):
                for j in range(nA):
                    if ncrysA == ncrysB:
                        C[icrys,i] += A[icrys,i,j]*b[icrys,j]
                    elif ncrysB == 1:
                        C[icrys,i] += A[icrys,i,j]*b[0,j]

    return C

@njit(float32[:,:,:](float32[:,:,:], float32[:,:,:]), fastmath=True, parallel=True)
def mat_mult(A, B):
    """
    Stacked matrix multiplication using numba on CPU.

    Parameters
    ----------
    A : ndarray
        (ncrys, n, n) array of matrices.
    B : ndarray
        (ncrys, n, n) array of matrices.

    Returns
    -------
    C : ndarray
        (ncrys, n, n) array, result of the matrix multiplication `A`*`B`.

    Examples
    --------
    >>> A = np.random.rand(1024,3,3).astype(DTYPEf)
    >>> B = np.random.rand(1024,3,3).astype(DTYPEf)
    >>> C1 = np.matmul(A,B)
    >>> C2 = mat_mult(A,B)
    >>> np.allclose(C1, C2, atol=1e-6)
    True
    """
    ncrysA, mA, nA = A.shape
    ncrysB, mB, nB = B.shape
    C = np.zeros((ncrysA,mA,nA), dtype=DTYPEf)
    if nA == mB:
        for icrys in prange(ncrysA):
            for i in range(mA):
                for j in range(nB):
                    for k in range(nA):
                        if ncrysA == ncrysB:
                            C[icrys,i,j] += A[icrys,i,k] * B[icrys,k,j]
                        elif ncrysB == 1:
                            C[icrys,i,j] += A[icrys,i,k] * B[0,k,j]
    return C

@njit(Tuple((float32[:,:], float32[:,:], int32[:]))(float32[:,:], int32, int32), fastmath=True, parallel=True)
def spherical_proj(vec, proj=0, north=3):
    """
    Performs stereographic or equal-area projection of vector `vec` in the equatorial plane.

    Parameters
    ----------
    vec : ndarray
        (ncrys, 3) array of unit vectors to be projected.
    proj : int, default=0
        type of projection, 0 for stereographic or 1 for equal-area.
    north : int, default=3
        North pole defining the projection plane.

    Returns
    -------
    xyproj : ndarray
        (ncrys, 2) array of projected coordinates in the equatorial plane.
    albeta : ndarray
        (ncrys, 2) array of [alpha, beta] polar angles in degrees.
    reverse : ndarray
        boolean array indicating where the input unit vectors were pointing to the Southern hemisphere and reversed.

    Examples
    --------
    >>> vec = np.random.rand(1024,3).astype(DTYPEf)
    >>> norm = np.sqrt(np.sum(vec**2, axis=1))
    >>> vec /= norm[..., np.newaxis]
    >>> xyproj0a, albeta0a, reverse0a = q4np.spherical_proj(vec, proj="stereo", north=3)
    >>> xyproj1a, albeta1a, reverse1a = q4np.spherical_proj(vec, proj="equal-area", north=3)
    >>> xyproj0b, albeta0b, reverse0b = spherical_proj(vec, proj=0, north=3)
    >>> xyproj1b, albeta1b, reverse1b = spherical_proj(vec, proj=1, north=3)
    >>> np.allclose(xyproj0a, xyproj0b, atol=1e-3)
    True
    >>> np.allclose(xyproj1a, xyproj1b, atol=1e-3)
    True
    >>> np.allclose(albeta0a, albeta0b, atol=0.5)
    True
    """
    sq2 = np.sqrt(2.)
    pi2 = 2.*np.pi
    if north == 1:
        x1 = 1; x2 = 2; x3 = 0
    elif north == 2:
        x1 = 2; x2 = 0; x3 = 1
    elif north == 3:
        x1 = 0; x2 = 1; x3 = 2
    else:
        x1 = 0; x2 = 1; x3 = 2

    ncrys = len(vec)
    albeta = np.zeros((ncrys,2), dtype=DTYPEf)
    xyproj = np.zeros((ncrys,2), dtype=DTYPEf)
    reverse = np.zeros(ncrys, dtype=DTYPEi)

    for j in prange(ncrys):
        v = vec[j,:]
        # check Northern hemisphere:
        if  v[x3] < -_EPS:
            v *= -1
            reverse[j] = 1

        # alpha:
        v[x3] = min(v[x3], 1.)
        v[x3] = max(v[x3],-1.)
        albeta[j,0] = np.arccos(v[x3])
        # beta:
        if np.abs(albeta[j,0]) > _EPS:
            tmp = v[x1]/np.sin(albeta[j,0])
            tmp = min(tmp, 1.)
            tmp = max(tmp,-1.)
            albeta[j,1] = np.arccos(tmp)
        if v[x2] < -_EPS:
            albeta[j,1] *= -1
        albeta[j,1] = albeta[j,1] % pi2

        xyproj[j,0] = np.cos(albeta[j,1])
        xyproj[j,1] = np.sin(albeta[j,1])
        if proj == 0: # stereographic projection
            Op = np.tan(albeta[j,0]/2.)
        else:                # equal-area projection
            Op = np.sin(albeta[j,0]/2.)*sq2
        xyproj[j,:] *= Op
        albeta[j,:] *= 360./pi2

    return xyproj, albeta, reverse

@njit(Tuple((float32[:,:], float32[:,:], float32[:,:], int32[:]))(float32[:,:], float32[:], float32[:,:], int32, int32), fastmath=True, parallel=True)
def q4_to_IPF(qarr, axis, qsym, proj=0, north=3):
    """
    Inverse Pole Figure projection based on crystal symmetries.

    Parameters
    ----------
    qarr : ndarray
        (ncrys, 4) array of quaternions.
    axis : array_like
        sample vector (in the reference frame) to be projected in the crystal IPF triangle.
    qsym : ndarray
        quaternion array of symmetry operations.
    proj : int, default=0
        type of projection, 0 for stereographic or 1 for equal-area.
    north : int, default=3
        North pole defining the projection plane.

    Returns
    -------
    xyproj : ndarray
        (ncrys, 2) array of projected coordinates in the standard triangle.
    RGB : ndarray
        (ncrys, 3) array of RGB color code for the projection.
    albeta : ndarray
        (ncrys, 2) array of [alpha, beta] polar angles in degrees.
    isym : ndarray
        index of the ith equivalent orientation corresponding to the standard triangle.

    Examples
    --------
    >>> qarr = q4np.q4_random(1024)
    >>> qsym = q4np.q4_sym_cubic()
    >>> axis = np.array([1,0,0], dtype=DTYPEf)
    >>> xyproj0, RGB0, albeta0, isym0 = q4np.q4_to_IPF(qarr, axis, qsym, proj="stereo", north=3)
    >>> xyproj1, RGB1, albeta1, isym1 = q4_to_IPF(qarr, axis, qsym, proj=0, north=3)
    >>> np.allclose(xyproj0, xyproj1, atol=1e-3)
    True
    >>> np.allclose(RGB0, RGB1, atol=0.001)
    True
    >>> np.allclose(albeta0, albeta1, atol=0.5)
    True
    >>> xyproj0, RGB0, albeta0, isym0 = q4np.q4_to_IPF(qarr, axis, qsym, proj="equal-area", north=3)
    >>> xyproj1, RGB1, albeta1, isym1 = q4_to_IPF(qarr, axis, qsym, proj=1, north=3)
    >>> np.allclose(xyproj0, xyproj1, atol=1e-3)
    True
    >>> np.allclose(RGB0, RGB1, atol=0.001)
    True
    >>> np.allclose(albeta0, albeta1, atol=0.5)
    True
    """
    deg2rad = np.pi/180.
    norm = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2 )
    axis /= norm
    bxis = np.zeros((1,3), dtype=DTYPEf)
    bxis[0,:] = axis
    ncrys = len(qarr)
    nsym = len(qsym)
    if nsym == 24:
        sym = 'cubic' if np.allclose(qsym, qsymCubic, atol=1e-6) else 'NA'
    elif nsym == 12:
        sym = 'hex'   if np.allclose(qsym, qsymHex, atol=1e-6) else 'NA'
    elif nsym == 8:
        sym = 'tetra' if np.allclose(qsym, qsymTetra, atol=1e-6) else 'NA'
    elif nsym == 4:
        sym = 'ortho' if np.allclose(qsym, qsymOrtho, atol=1e-6) else 'NA'
    elif nsym == 2:
        sym = 'mono'  if np.allclose(qsym, qsymMono, atol=1e-6) else 'NA'
    else:
        sym = 'unknown'

    albeta = np.zeros((ncrys,2), dtype=DTYPEf)+360
    xyproj = np.zeros((ncrys,2), dtype=DTYPEf)
    isym   = np.zeros(ncrys, dtype=DTYPEi)

    for iq, q in enumerate(qsym):
        qequ = q4_mult(qarr, qsym[iq:iq+1])
        Rsa2cr = q4_to_mat(qequ)
        #vec = np.dot(Rsa2cr, axis)
        vec = matvec_mult(Rsa2cr, bxis)

        xyproj1, albeta1, reverse = spherical_proj(vec, proj=proj, north=north)
        for j in prange(ncrys):
            update = False
            if sym == 'cubic':
                if (albeta1[j,1] < 45.+_EPS) & (albeta1[j,0] < albeta[j,0] +_EPS):
                    update = True
            elif sym == 'hex':
                if (albeta1[j,1] < 30.+_EPS):
                    update = True
            elif sym == 'tetra':
                if (albeta1[j,1] < 45.+_EPS):
                    update = True
            elif sym == 'ortho':
                if (albeta1[j,1] < 90.+_EPS):
                    update = True
            elif sym == 'mono':
                if (albeta1[j,1] < 180.+_EPS):
                    update = True
            else:
                if (albeta1[j,1] > -_EPS):
                    update = True

            if update:
                xyproj[j,:] = xyproj1[j,:]
                albeta[j,:] = albeta1[j,:]
                isym[j] = iq

    RGB = np.zeros((ncrys,3), dtype=DTYPEf)
    for j in prange(ncrys):
        if sym == 'cubic':
            alpha_max = np.arccos(np.sqrt(1./(2.+np.tan(albeta[j,1]*deg2rad)**2)))/deg2rad
            beta_max = 45.
        elif sym == 'hex':
            alpha_max = 90.
            beta_max = 30.
        elif sym == 'tetra':
            alpha_max = 90.
            beta_max = 45.
        elif sym == 'ortho':
            alpha_max = 90.
            beta_max = 90.
        elif sym == 'mono':
            alpha_max = 90.
            beta_max = 180.
        else:
            alpha_max = 90.
            beta_max = 360.

        RGB[j,0] =  1. - albeta[j,0]/alpha_max
        RGB[j,1] = (1. - albeta[j,1]/beta_max) * albeta[j,0]/alpha_max
        RGB[j,2] = (albeta[j,1]/beta_max)      * albeta[j,0]/alpha_max
        mx = RGB[j,:].max()
        RGB[j,:] /= mx

    return xyproj, RGB, albeta, isym

#@njit(Tuple((float32[:], float32[:], float32[:], float32[:]))(float32[:,:], float32[:,:]), fastmath=True, parallel=True)
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
    >>> ang = q4np.q4_angle(qa[0], qavg)
    >>> np.allclose(ang, 0., atol=0.5)
    True
    >>> qavg2, GROD2, GROD_stat2, theta_iter2 = q4np.q4_mean_disori(qarr, qsym)
    >>> ang2 = q4np.q4_angle(qavg, qavg2)
    >>> np.allclose(ang2, 0., atol=0.5)
    True
    >>> np.allclose(GROD, GROD2, atol=0.5)
    True
    """

    ii = 0
    theta= 999.
    mxtheta = 0.2
    nitermax = 10
    theta_iter = np.zeros(nitermax, dtype=DTYPEf) - 1.
    ncrys = qarr.shape[0]

    # initialize avg orientation:
    #qref = qarr[0:1,:]
    qmed = np.atleast_2d(np.median(qarr, axis=0))
    cosang = np.minimum(np.abs( qmed[0,0]*qarr[:,0] +
                                qmed[0,1]*qarr[:,1] +
                                qmed[0,2]*qarr[:,2] +
                                qmed[0,3]*qarr[:,3]), 1.)
    imed = np.argmax(cosang)
    qref = qarr[imed:imed+1,:]
    while (theta > mxtheta) and (ii < nitermax):
        # disorientation of each crystal wrt average orientation:
        qdis, _ = q4_disori_quat(qref, qarr, qsym, frame=1, method=1)

        qtmp = np.zeros((1,4), dtype=np.float64)
        #qtmp[0] = np.sum(qdis, axis=0) # careful with np.float32 sum of very big arrays with more than 16*1024**2 quaternions
        qtmp[0] = np.mean(qdis, axis=0, dtype=np.float64)
        qtmp /= np.sqrt(qtmp[0,0]**2 + qtmp[0,1]**2 + qtmp[0,2]**2 + qtmp[0,3]**2)
        qtmp = qtmp.astype(np.float32)

        # q_mean=q_ref*q_sum/|q_sum|
        qavg = q4_mult(qref, qtmp)

        #### theta for convergence of qavg:
        theta = np.arccos(q4_cosang2(qref, qavg)[0])*2*180/np.pi
        theta_iter[ii] = theta
        qref = qavg

        ii += 1

    # angles:
    GROD = np.minimum(np.abs(qdis[:,0]), 1.)
    GROD = np.arccos(GROD, dtype=DTYPEf)*2*180/np.pi
    #GROD = np.zeros(ncrys, dtype=DTYPEf)
    #for j in prange(ncrys):
    #    GROD[j] = np.arccos(min(abs(qdis[j,0]), 1.))*2*180/np.pi

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

def q4_mean_multigrain(qarr, qsym, unigrain, iunic, iback):
    """
    Average orientation and disorientation (multigrain).

    Parameters
    ----------
    qarr : ndarray
        (ncrys, 4) array of quaternions.
    qsym : ndarray
        quaternion array of symmetry operations.
    unigrain : ndarray
        (nlab,) array of unic grain labels.
    iunic : ndarray
        (nlab,) array of indices pointing to the first occurence of each grain label in the complete (ncrys,) array.
    iback : ndarray
        (ncrys,) array of indices allowing to reconstruct the complete (ncrys,...) arrays from the grain averages (nlab,...) arrays.

    Returns
    -------
    qavg_unic : ndarray
        (nlab, 4) array of grain average quaternions.
    GOS_unic : ndarray
        (nlab,) array of grain orientation spread.
    theta_unic : ndarray
        (nlab,) array of final grain convergence angle.
    GROD : ndarray
        (ncrys,) array of grain reference orientation deviation in degrees.
    theta_iter : ndarray
        convergence angle (degree) during the iterations for `qavg`.

    Examples
    --------
    >>> qa = q4np.q4_random(256)
    >>> grains = np.repeat(np.arange(0,256), 1024) + 1
    >>> np.random.shuffle(grains)
    >>> unic, iunic, iback = np.unique(grains, return_index=True, return_inverse=True)
    >>> qarr = q4_mult(qa[grains - 1], q4np.q4_orispread(ncrys=1024*256, thetamax=2., misori=True))
    >>> qsym = q4np.q4_sym_cubic()
    >>> qavg, GOS, theta, GROD, theta_iter = q4_mean_multigrain(qarr, qsym, unic, iunic, iback)
    >>> ang = q4np.q4_angle(qa, qavg)
    >>> np.allclose(ang, np.zeros_like(ang), atol=0.5)
    True
    """

    ii = 0
    mxtheta = 0.2
    nitermax = 3
    theta_iter = np.zeros(nitermax, dtype=DTYPEf) - 1.

    grains = unigrain[iback]

    #unic, iunic, iback, counts = np.unique(grains, return_index=True, return_inverse=True, return_counts=True)
    theta_unic = np.zeros(len(unigrain), dtype=DTYPEf) + 999.
    theta = theta_unic[iback]
    qdis = np.zeros_like(qarr)

    # update iunic to account for the median quaternion in each grain, instead of the first, to initialize the average loop:
    qmed = np.zeros((len(unigrain), 4), dtype=DTYPEf)
    qmed[:,0] = ndi.median(qarr[:,0], grains, index=unigrain)
    qmed[:,1] = ndi.median(qarr[:,1], grains, index=unigrain)
    qmed[:,2] = ndi.median(qarr[:,2], grains, index=unigrain)
    qmed[:,3] = ndi.median(qarr[:,3], grains, index=unigrain)
    qmed = qmed[iback] # back to full size ncrys
    cosang = q4_cosang2(qmed, qarr)

    imed = ndi.maximum_position(cosang, grains, index=unigrain)
    imed = np.squeeze(np.array(imed, dtype=DTYPEi))
    qref_unic = qarr[imed]
    #qref_unic = qarr[iunic]

    while (theta_unic.max() > mxtheta) and (ii < nitermax):
        qref_tot = qref_unic[iback]

        # disorientation of each crystal wrt average orientation:
        whrT = (theta > mxtheta)
        qdis[whrT], jj = q4_disori_quat(qref_tot[whrT], qarr[whrT], qsym, frame=1, method=1)

        qdis_unic = np.zeros(qref_unic.shape, dtype=np.float64)
        qdis_unic[:,0] = ndi.mean(qdis[:,0], grains, index=unigrain) # careful with np.float32 sum of very big arrays with more than 16*1024**2 quaternions
        qdis_unic[:,1] = ndi.mean(qdis[:,1], grains, index=unigrain)
        qdis_unic[:,2] = ndi.mean(qdis[:,2], grains, index=unigrain)
        qdis_unic[:,3] = ndi.mean(qdis[:,3], grains, index=unigrain)

        norm = np.sqrt(np.einsum('...i,...i', qdis_unic, qdis_unic))
        qdis_unic /= norm[..., np.newaxis]
        qdis_unic = qdis_unic.astype(np.float32)

        # q_mean=q_ref*q_sum/|q_sum|
        qavg_unic = q4_mult(qref_unic, qdis_unic)

        #### theta for convergence of qavg:
        theta_unic = np.arccos(q4_cosang2(qref_unic, qavg_unic))*2*180/np.pi
        theta = theta_unic[iback]
        theta_iter[ii] = theta_unic.max()
        qref_unic = qavg_unic

        ii += 1

    # angles:
    GROD = np.minimum(np.abs(qdis[:,0]), 1.)
    GROD = np.arccos(GROD)*2*180/np.pi

    GOS_unic = ndi.mean(GROD, grains, index=unigrain)

    theta_iter = theta_iter[theta_iter >= 0.]

    logging.info("Computed average grain orientation over {} crystals and {} grains in {} iterations.".format(len(qarr), len(unigrain), ii))
    logging.info("Theta convergence (degrees): {}".format(theta_iter))

    return qavg_unic, GOS_unic, theta_unic, GROD, theta_iter

if __name__ == "__main__":
    import doctest
    doctest.testmod()

