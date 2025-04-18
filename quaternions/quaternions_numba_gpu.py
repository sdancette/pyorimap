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
import cupyx.scipy.ndimage as ndix
from pyorimap.quaternions import quaternions_np as q4np

from numba import cuda, uint8, int32, float32, boolean
from numba.types import Tuple

DTYPEf = np.float32
DTYPEi = np.int32
_EPS = 1e-7
sq2 = np.sqrt(2.)
pi2 = 2.*np.pi

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

@cuda.jit((float32[:], float32), fastmath=True, device=True)
def _device_q4_positive(qa, _EPS):
    opposite = False
    if (qa[0] < -_EPS):
        opposite = True
    elif (qa[0] >= -_EPS)*(qa[0] < _EPS):
        if (qa[1] < -_EPS):
            opposite = True
        elif (qa[1] >= -_EPS)*(qa[1] < _EPS):
            if (qa[2] < -_EPS):
                opposite = True
            elif (qa[2] >= -_EPS)*(qa[2] < _EPS):
                if (qa[3] < -_EPS):
                    opposite = True

    if opposite:
        qa[0] = -qa[0]
        qa[1] = -qa[1]
        qa[2] = -qa[2]
        qa[3] = -qa[3]

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

@cuda.jit((float32[:], float32[:,:]), fastmath=True, device=True)
def _device_q4_to_mat(q, R):
    q0 = q[0]; q1 = q[1]; q2 = q[2]; q3 = q[3]
    R[0,0] = 2.*(q0**2+q1**2-1./2)
    R[1,0] = 2.*(q1*q2-q0*q3)
    R[2,0] = 2.*(q1*q3+q0*q2)
    R[0,1] = 2.*(q1*q2+q0*q3)
    R[1,1] = 2.*(q0**2+q2**2-1./2)
    R[2,1] = 2.*(q2*q3-q0*q1)
    R[0,2] = 2.*(q1*q3-q0*q2)
    R[1,2] = 2.*(q2*q3+q0*q1)
    R[2,2] = 2.*(q0**2+q3**2-1./2)

@cuda.jit((float32[:,:], float32[:], float32[:]), fastmath=True, device=True)
def _device_matvec_mult(A,b,c):
    c[0] = A[0,0]*b[0] + A[0,1]*b[1] + A[0,2]*b[2]
    c[1] = A[1,0]*b[0] + A[1,1]*b[1] + A[1,2]*b[2]
    c[2] = A[2,0]*b[0] + A[2,1]*b[1] + A[2,2]*b[2]
    #m, n = A.shape
    #for i in range(m):
    #    for j in range(n):
    #        c[i] += A[i,j]*b[j]

@cuda.jit(float32(float32[:], float32[:]), fastmath=True, device=True)
def _device_q4_cosang2(qa, qb):
    return( min(abs(qa[0]*qb[0] + qa[1]*qb[1] + qa[2]*qb[2] + qa[3]*qb[3]), 1.) )

@cuda.jit((float32[:], int32, int32, float32[:], float32[:], boolean), fastmath=True, device=True)
def _device_spherical_proj(vec, proj, north, xyproj, albeta, reverse):
    if north == 1:
        x1 = 1; x2 = 2; x3 = 0
    elif north == 2:
        x1 = 2; x2 = 0; x3 = 1
    elif north == 3:
        x1 = 0; x2 = 1; x3 = 2
    else:
        x1 = 0; x2 = 1; x3 = 2

    reverse = False
    # check Northern hemisphere:
    if  vec[x3] < -_EPS:
        reverse = True
        vec[x1] *= -1.
        vec[x2] *= -1.
        vec[x3] *= -1.

    # alpha:
    vec[x3] = min(vec[x3], 1.)
    vec[x3] = max(vec[x3],-1.)
    albeta[0] = cuda.libdevice.acosf(vec[x3])
    # beta:
    if abs(albeta[0]) > _EPS:
        tmp = vec[x1]/cuda.libdevice.sinf(albeta[0])
        tmp = min(tmp, 1.)
        tmp = max(tmp,-1.)
        albeta[1] = cuda.libdevice.acosf(tmp)
    if vec[x2] < -_EPS:
        albeta[1] *= -1
    albeta[1] = albeta[1] % pi2

    xyproj[0] = cuda.libdevice.cosf(albeta[1])
    xyproj[1] = cuda.libdevice.sinf(albeta[1])
    if proj == 0: # stereographic projection
        Op = cuda.libdevice.tanf(albeta[0]/2.)
    else:                # equal-area projection
        Op = cuda.libdevice.sinf(albeta[0]/2.)*sq2
    xyproj[0] *= Op
    xyproj[1] *= Op
    albeta[0] *= 360./pi2
    albeta[1] *= 360./pi2

@cuda.jit((float32[:,:], int32, int32, float32[:,:], float32[:,:], boolean[:]))
def _kernel_spherical_proj(vec, proj, north, xyproj, albeta, reverse):
    i = cuda.grid(1)
    if i < vec.shape[0]:
        _device_spherical_proj(vec[i,:], proj, north, xyproj[i,:], albeta[i,:], reverse[i])

@cuda.jit((float32[:,:], float32[:], float32[:,:], int32, int32, float32[:,:], float32[:,:], uint8[:]))
def _kernel_IPF_proj(qa, axis, qsym, proj, north, xyproj, albeta, isym):
    i = cuda.grid(1)
    if i < qa.shape[0]:
        # initialize arrays:
        qc = cuda.local.array(shape=4, dtype=float32)
        Rsa2cr = cuda.local.array(shape=(3,3), dtype=float32)
        vec = cuda.local.array(shape=3, dtype=float32)
        xyproj1 = cuda.local.array(shape=2, dtype=float32)
        albeta1 = cuda.local.array(shape=2, dtype=float32)
        reverse = False
        nsym = len(qsym)

        # loop on symmetries:
        for iq in range(nsym):
            _device_q4_mult(qa[i,:], qsym[iq,:], qc)
            _device_q4_to_mat(qc, Rsa2cr)
            _device_matvec_mult(Rsa2cr,axis,vec)

            _device_spherical_proj(vec, proj, north, xyproj1, albeta1, reverse)

            update = False
            if nsym == 24: #'cubic'
                if (albeta1[1] < 45.+_EPS)*(albeta1[0] < albeta[i,0]+_EPS):
                    update = True
            elif nsym == 12: #'hex'
                if (albeta1[1] < 30.+_EPS):
                    update = True
            elif nsym == 8: #'tetra'
                if (albeta1[1] < 45.+_EPS):
                    update = True
            elif nsym == 4: #'ortho'
                if (albeta1[1] < 90.+_EPS):
                    update = True
            elif nsym == 2: #'mono'
                if (albeta1[1] < 180.+_EPS):
                    update = True
            else:
                if (albeta1[1] > -_EPS):
                    update = True

            if update:
                xyproj[i,0] = xyproj1[0]; xyproj[i,1] = xyproj1[1]
                albeta[i,0] = albeta1[0]; albeta[i,1] = albeta1[1]
                isym[i] = iq

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

@cuda.jit((float32[:,:], float32[:,:,:]))
def _kernel_q4_to_mat(q, R):
    i = cuda.grid(1)
    if i < q.shape[0]:
        _device_q4_to_mat(q[i,:], R[i,:,:])

@cuda.jit((float32[:,:,:], float32[:,:], float32[:,:]))
def _kernel_matvec_mult(A,b,c):
    i = cuda.grid(1)
    if i < A.shape[0]:
        _device_matvec_mult(A[i,:,:], b[i,:], c[i,:])

@cuda.jit((float32[:,:], float32[:,:], float32[:]))
def _kernel_q4_cosang2(qa, qb, cosang):
    i = cuda.grid(1)
    if i < qa.shape[0]:
        cosang[i] = _device_q4_cosang2(qa[i,:], qb[i,:])

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
            if b1 > b0:
                b0 = b1

        ang[i] = np.arccos(b0) * 2 * 180/np.pi

@cuda.jit((float32[:,:], float32[:,:], float32[:,:], float32[:,:], int32), fastmath=True)
def _kernel_q4_disori_quat(qa, qb, qsym, qdis, frame):
    i = cuda.grid(1)
    if i < qa.shape[0]:
        # initialize arrays:
        b0 = 0.
        qa_inv = cuda.local.array(shape=4, dtype=float32)
        qc = cuda.local.array(shape=4, dtype=float32)

        ### identify the i_th symmetry corresponding to the smallest misorientation:
        # disorientation qc expressed in the frame of crystal a:
        _device_q4_inv(qa[i,:], qa_inv)
        _device_q4_mult(qa_inv, qb[i,:], qc)

        isym = 0
        for j in range(len(qsym)):
            b1 = _device_q4_cosang2(qc, qsym[j,:])
            if b1 > b0:
                b0 = b1
                isym = j

        # express disorientation quaternion with i_th symmetry:
        _device_q4_mult(qa[i,:], qsym[isym,:], qc)
        _device_q4_inv(qc, qa_inv)
        if frame == 0:
            _device_q4_mult(qb[i,:], qa_inv, qdis[i,:])
        else:
            _device_q4_mult(qa_inv, qb[i,:], qdis[i,:])
        _device_q4_positive(qdis[i,:], _EPS)

@cuda.jit((float32[:,:], float32[:,:], float32[:,:]), fastmath=True)
def _kernel_q4_to_FZ(qa, qsym, qFZ):
    i = cuda.grid(1)
    if i < qa.shape[0]:
        # initialize arrays:
        b0 = 0.
        qc = cuda.local.array(shape=4, dtype=float32)

        for j in range(len(qsym)):
            _device_q4_mult(qa[i,:], qsym[j,:], qc)
            b1 = min(abs(qc[0]), 1.)
            if b1 > b0:
                b0 = b1
                _device_q4_positive(qc, _EPS)
                qFZ[i,0] = qc[0]
                qFZ[i,1] = qc[1]
                qFZ[i,2] = qc[2]
                qFZ[i,3] = qc[3]

def spherical_proj(vec, proj, north, xyproj, albeta, reverse, nthreads=256):
    """
    Wrapper calling the kernel to compute spherical projection.

    Numba GPU version, all input arrays already on GPU memory.

    Parameters
    ----------
    vec : ndarray
        unit vectors to be projected, on GPU memory.

    Examples
    --------
    >>> qa = q4np.q4_random(1024)
    """
    threadsperblock = nthreads
    blockspergrid = (vec.shape[0] + (threadsperblock - 1)) // threadsperblock
    _kernel_spherical_proj[blockspergrid, threadsperblock](vec, proj, north, xyproj, albeta, reverse)

def q4_to_IPF_proj(qa, axis, qsym, proj, north, xyproj, albeta, isym, nthreads=256):
    """
    IPF projection on GPU.
    """
    threadsperblock = nthreads
    blockspergrid = (qa.shape[0] + (threadsperblock - 1)) // threadsperblock
    # initialize alpha-beta for the search of fundamental zone:
    albeta *= 0.
    albeta += 360.
    # normalize axis:
    norm = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2 )
    axis /= norm

    _kernel_IPF_proj[blockspergrid, threadsperblock](qa, axis, qsym, proj, north, xyproj, albeta, isym)

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

def q4_to_mat(q, R, nthreads=256):
    """
    Wrapper calling the kernel to compute the rotation matrix from unit quaternions.

    Numba GPU version, all input arrays already on GPU memory.

    Parameters
    ----------
    q : ndarray
        array of quaternions on GPU memory.
    R : ndarray
        corresponding array of rotation matrices, modified in place on GPU memory.

    Examples
    --------
    >>> qarr = q4np.q4_random(1024)
    >>> R0 = q4np.q4_to_mat(qarr)
    >>> qarrGPU = cp.asarray(qarr)
    >>> RGPU = cp.zeros(R0.shape, dtype=DTYPEf)
    >>> q4_to_mat(qarrGPU, RGPU)
    >>> np.allclose(R0, RGPU.get(), atol=1e-6)
    True
    """
    threadsperblock = nthreads
    blockspergrid = (q.shape[0] + (threadsperblock - 1)) // threadsperblock
    _kernel_q4_to_mat[blockspergrid, threadsperblock](q, R)

def matvec_mult(A, b, c, nthreads=256):
    """
    Wrapper calling the kernel to compute matrix-vector multiplication.

    Numba GPU version, all input arrays already on GPU memory.

    Parameters
    ----------
    A : ndarray
        stacked (n,3,3) matrices on GPU memory.
    b : ndarray
        stacked (n,3) vectors on GPU memory.
    c : ndarray
        result of `A`*`b`, modified in place on GPU memory.

    Examples
    --------
    >>> A = np.random.rand(1024, 3,3).astype(DTYPEf)
    >>> b = np.random.rand(1024, 3).astype(DTYPEf)
    >>> c = np.matvec(A,b)
    >>> AGPU = cp.asarray(A)
    >>> bGPU = cp.asarray(b)
    >>> cGPU = cp.zeros_like(c)
    >>> matvec_mult(AGPU, bGPU, cGPU)
    >>> np.allclose(c, cGPU.get(), atol=1e-6)
    True
    """
    threadsperblock = nthreads
    blockspergrid = (A.shape[0] + (threadsperblock - 1)) // threadsperblock
    _kernel_matvec_mult[blockspergrid, threadsperblock](A, b, c)

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

def q4_cosang2(qa, qb, cosang, nthreads=256):
    """
    Wrapper calling the kernel to compute the cosine of the half angle between `qa` and `qb`.

    Numba GPU version, all input arrays already on GPU memory.

    Parameters
    ----------
    qa : ndarray
        array of quaternions or single quaternion on GPU memory.
    qb : ndarray
        array of quaternions or single quaternion on GPU memory.
    cosang : ndarray
        the cosine of the half angle between quaternions `qa` and `qb`. on GPU memory.

    Examples
    --------
    >>> qa = q4np.q4_random(1024)
    >>> ang = np.random.rand(1024)*180.
    >>> qrot = q4np.q4_from_axis_angle(np.random.rand(1024,3), ang)
    >>> qb = q4np.q4_mult(qa, qrot)
    >>> qa_gpu = cp.asarray(qa, dtype=DTYPEf)
    >>> qb_gpu = cp.asarray(qb, dtype=DTYPEf)
    >>> cosang_gpu = cp.zeros_like(qa_gpu[:,0])
    >>> q4_cosang2(qa_gpu, qb_gpu, cosang_gpu, nthreads=128)
    >>> aback = np.degrees(2*np.arccos(cp.asnumpy(cosang_gpu)))
    >>> np.allclose(aback, ang, atol=0.1)
    True
    """
    threadsperblock = nthreads
    blockspergrid = (qa.shape[0] + (threadsperblock - 1)) // threadsperblock
    _kernel_q4_cosang2[blockspergrid, threadsperblock](qa, qb, cosang)

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
    >>> np.allclose(ang, cp.asnumpy(ang_gpu), atol=0.1)
    True
    """
    threadsperblock = nthreads
    blockspergrid = (qa.shape[0] + (threadsperblock - 1)) // threadsperblock
    _kernel_q4_disori_angle[blockspergrid, threadsperblock](qa, qb, qsym, ang)

def q4_disori_quat(qa, qb, qsym, qdis, frame=0, nthreads=256):
    """
    Wrapper calling the kernel to compute disorientation quaternion between
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
    qdis : ndarray
        disorientation quaternion between quaternions `qa` and `qb`,
        taking symmetries into account, modified in place on GPU memory.
    frame : int, default=0
        the frame to express the disorientation quaternion, 0='ref' or 1='crys_a'.

    Examples
    --------
    >>> qa = q4np.q4_random(1024)
    >>> qb = qa[::-1,:]
    >>> qsym = q4np.q4_sym_cubic()
    >>> qdis = q4np.q4_disori_quat(qa, qb, qsym, frame='ref')
    >>> qa_gpu = cp.asarray(qa, dtype=DTYPEf)
    >>> qb_gpu = cp.asarray(qb, dtype=DTYPEf)
    >>> qsym_gpu = cp.asarray(qsym, dtype=DTYPEf)
    >>> qdis_gpu = cp.zeros_like(qa_gpu)
    >>> q4_disori_quat(qa_gpu, qb_gpu, qsym_gpu, qdis_gpu, frame=0, nthreads=128)
    >>> np.allclose(qdis, q4np.q4_positive(cp.asnumpy(qdis_gpu)), atol=1e-6)
    True
    >>> qdis = q4np.q4_disori_quat(qa, qb, qsym, frame='crys_a')
    >>> q4_disori_quat(qa_gpu, qb_gpu, qsym_gpu, qdis_gpu, frame=1, nthreads=128)
    >>> np.allclose(qdis, q4np.q4_positive(cp.asnumpy(qdis_gpu)), atol=1e-6)
    True
    """
    threadsperblock = nthreads
    blockspergrid = (qa.shape[0] + (threadsperblock - 1)) // threadsperblock
    _kernel_q4_disori_quat[blockspergrid, threadsperblock](qa, qb, qsym, qdis, frame)

def q4_to_FZ(qa, qsym, qFZ, nthreads=256):
    """
    Wrapper calling the kernel to move quaternions to Fundamental Zone.

    Numba GPU version, all input arrays already on GPU memory.

    Parameters
    ----------
    qa : ndarray
        array of quaternions or single quaternion on GPU memory.
    qsym : ndarray
        quaternion array of symmetry operations on GPU memory.
    qFZ : ndarray
        array of quaternions in the Fundamental Zone, modified in place on GPU memory.

    Examples
    --------
    >>> qsym = q4np.q4_sym_cubic()
    >>> qsym_gpu = cp.asarray(qsym, dtype=DTYPEf)
    >>> qFZ_gpu = cp.zeros_like(qsym_gpu)
    >>> q4_to_FZ(qsym_gpu, qsym_gpu, qFZ_gpu)
    >>> ref_gpu = cp.tile(cp.array([1,0,0,0], dtype=DTYPEf), 24).reshape(24, -1)
    >>> cp.allclose(qFZ_gpu, ref_gpu, atol=1e-6)
    array(True)
    >>> qa = q4np.q4_random(1024)
    >>> qa_gpu = cp.asarray(qa, dtype=DTYPEf)
    >>> qFZ_gpu = cp.zeros_like(qa_gpu)
    >>> q4_to_FZ(qa_gpu, qsym_gpu, qFZ_gpu)
    >>> qFZ2 = q4np.q4_to_FZ(qa, qsym, return_index=False)
    >>> np.allclose(qFZ2, cp.asnumpy(qFZ_gpu), atol=1e-6)
    True
    """
    threadsperblock = nthreads
    blockspergrid = (qa.shape[0] + (threadsperblock - 1)) // threadsperblock
    _kernel_q4_to_FZ[blockspergrid, threadsperblock](qa, qsym, qFZ)

def q4_mean_disori(qarr, qsym, qavg, GROD, GROD_stat, theta_iter):
    """
    Average orientation and disorientation (GOS and GROD).

    Numba GPU version, all input arrays already on GPU memory.

    Parameters
    ----------
    qarr : ndarray
        (n, 4) array of quaternions on GPU memory.
    qsym : ndarray
        quaternion array of symmetry operations on GPU memory.
    qavg : ndarray
        quaternion representing the average orientation of `qarr`, modified in place on GPU memory.
    GROD : ndarray
        (n,) array of grain reference orientation deviation in degrees, modified in place on GPU memory.
    GROD_stat : ndarray
        [mean, std, min, Q1, median, Q3, max] of the grain reference orientation deviation (GROD), modified in place on GPU memory.
        GROD_stat[0] is the grain orientation spread (GOS), i.e. the average disorientation angle in degrees.
    theta_iter : ndarray
        convergence angle (degree) during the iterations for `qavg`, modified in place on GPU memory.

    Examples
    --------
    >>> qa = q4np.q4_random(1)
    >>> qarr = cp.asarray(q4np.q4_mult(qa, q4np.q4_orispread(ncrys=1024, thetamax=2., misori=True)), dtype=DTYPEf)
    >>> qsym = cp.asarray(q4np.q4_sym_cubic())
    >>> qavg = cp.zeros((1,4), dtype=DTYPEf)
    >>> GROD = cp.zeros(1024, dtype=DTYPEf)
    >>> GROD_stat = cp.zeros(7, dtype=DTYPEf)
    >>> theta_iter = cp.zeros(10, dtype=DTYPEf)
    >>> q4_mean_disori(qarr, qsym, qavg, GROD, GROD_stat, theta_iter)
    >>> ang = q4np.q4_angle(qa, cp.asnumpy(qavg[0]))
    >>> np.allclose(ang, 0., atol=0.5)
    True
    >>> qavg2, GROD2, GROD_stat2, theta_iter2 = q4np.q4_mean_disori(cp.asnumpy(qarr), cp.asnumpy(qsym))
    >>> ang2 = q4np.q4_angle(cp.asnumpy(qavg[0]), qavg2)
    >>> np.allclose(ang2, 0., atol=0.5)
    True
    >>> np.allclose(cp.asnumpy(GROD), GROD2, atol=0.5)
    True
    """

    ii = 0
    theta= 999.
    mxtheta = 0.2
    nitermax = 10
    theta_iter -= 1.
    ncrys = qarr.shape[0]

    qdis = cp.zeros_like(qarr)
    #qavg = cp.zeros((1,4), dtype=DTYPEf)

    # initialize avg orientation:
    #qref = qarr[0:1,:]
    qmed = cp.atleast_2d(cp.median(qarr, axis=0))
    q4_cosang2(qmed, qarr, GROD)
    #cp.minimum(cp.abs(qmed[0,0]*qarr[:,0] +
    #                  qmed[0,1]*qarr[:,1] +
    #                  qmed[0,2]*qarr[:,2] +
    #                  qmed[0,3]*qarr[:,3]), 1., out=GROD)
    imed = cp.argmax(GROD)
    qref = qarr[imed:imed+1,:]

    while (theta > mxtheta) and (ii < nitermax):
        # disorientation of each crystal wrt average orientation:
        qdis *= 0.
        #print(qref)
        q4_disori_quat(cp.tile(qref[0], len(qarr)).reshape(-1,4),
                       qarr, qsym, qdis, frame=1, nthreads=256) ## check kernel with unbalanced shape of qa and qb in q4_disori !

        #qtmp = cp.sum(qdis, axis=0) # careful with np.float32 sum of very big arrays with more than 16*1024**2 quaternions
        #qtmp /= cp.sqrt(cp.einsum('...i,...i', qtmp, qtmp))
        #qtmp /= cp.sqrt(cp.sum(qtmp**2)) # slower ?
        qtmp = cp.mean(qdis, axis=0, dtype=cp.float64)
        qtmp /= cp.sqrt(qtmp[0]**2 + qtmp[1]**2 + qtmp[2]**2 + qtmp[3]**2)
        qtmp = qtmp.astype(cp.float32)

        # q_mean=q_ref*q_sum/|q_sum|
        q4_mult(qref, cp.atleast_2d(qtmp).astype(cp.float32), qavg)

        #### theta for convergence of qavg:
        theta = cp.arccos( min(abs(qref[0,0]*qavg[0,0] +
                                   qref[0,1]*qavg[0,1] +
                                   qref[0,2]*qavg[0,2] +
                                   qref[0,3]*qavg[0,3]), 1.), dtype=DTYPEf)*2*180/cp.pi
        theta_iter[ii] = theta
        qref = qavg
        ii += 1

    # angles:
    GROD[:] = cp.arccos(cp.minimum(cp.abs(qdis[:,0]), 1.))*2*180/cp.pi

    GOS = GROD.mean()
    GROD_stat[0] = GOS
    GROD_stat[1] = GROD.std()
    GROD_stat[2] = GROD.min()
    GROD_stat[3] = cp.quantile(GROD, 0.25)
    GROD_stat[4] = cp.median(GROD)
    GROD_stat[5] = cp.quantile(GROD, 0.75)
    GROD_stat[6] = GROD.max()

def q4_mean_multigrain(qarr, qsym, unigrain, iunic, iback):
    """
    Average orientation and disorientation (multigrain).

    Numba GPU version, all input arrays already on GPU memory.

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
    >>> qa = q4np.q4_random(100)
    >>> grains = np.repeat(np.arange(0,100), 1024) + 1
    >>> np.random.shuffle(grains)
    >>> unic, iunic, iback = np.unique(grains, return_index=True, return_inverse=True)
    >>> qarr = q4np.q4_mult(qa[grains - 1], q4np.q4_orispread(ncrys=1024*100, thetamax=2., misori=True))
    >>> qsym = q4np.q4_sym_cubic()
    >>> qarr_gpu = cp.asarray(qarr)
    >>> qsym_gpu = cp.asarray(qsym)
    >>> unic_gpu =  cp.asarray(unic.astype(DTYPEi))
    >>> iunic_gpu = cp.asarray(iunic.astype(DTYPEi))
    >>> iback_gpu = cp.asarray(iback.astype(DTYPEi))
    >>> qavg_gpu, GOS_gpu, theta_gpu, GROD_gpu, theta_iter_gpu = q4_mean_multigrain(qarr_gpu, qsym_gpu, unic_gpu, iunic_gpu, iback_gpu)
    >>> ang = q4np.q4_angle(qa, cp.asnumpy(qavg_gpu))
    >>> np.allclose(ang, np.zeros_like(ang), atol=0.5)
    True
    """

    ii = 0
    mxtheta = 0.2
    nitermax = 3
    theta_iter = cp.zeros(nitermax, dtype=DTYPEf) - 1.

    grains = unigrain[iback]
    GROD = cp.zeros(grains.size, dtype=cp.float32)

    theta_unic = cp.zeros(len(unigrain), dtype=DTYPEf) + 999.
    qavg_unic = cp.zeros((len(unigrain), 4), dtype=cp.float32)
    qdis_unic = cp.zeros((len(unigrain), 4), dtype=cp.float64)

    # update iunic to account for the median quaternion in each grain, instead of the first, to initialize the average loop:
    qmed = cp.zeros((len(unigrain), 4), dtype=DTYPEf)
    qmed[:,0] = ndix.median(qarr[:,0], grains, index=unigrain)
    qmed[:,1] = ndix.median(qarr[:,1], grains, index=unigrain)
    qmed[:,2] = ndix.median(qarr[:,2], grains, index=unigrain)
    qmed[:,3] = ndix.median(qarr[:,3], grains, index=unigrain)
    qmed = qmed[iback] # temporarily stores qmed back to full size ncrys
    q4_cosang2(qmed, qarr, GROD) # temporarily stores cosine in GROD

    # free memory before starting the loop:
    qmed = qmed[0:1]*0.
    mempool.free_all_blocks()

    imed = ndix.maximum_position(GROD, grains, index=unigrain)
    imed = cp.squeeze(cp.array(imed, dtype=DTYPEi))
    qref_unic = qarr[imed]
    #qref_unic = qarr[iunic]

    while (theta_unic.max() > mxtheta) and (ii < nitermax):
        qdis = qref_unic[iback] # temporarily stores reference orientation before update

        # disorientation of each crystal wrt average orientation:
        q4_disori_quat(qdis, qarr, qsym,
                       qdis, frame=1, nthreads=256)

        cp.minimum(cp.abs(qdis[:,0]), 1., out=GROD)

        qdis_unic *= 0.
        qdis_unic[:,0] = ndix.mean(qdis[:,0], grains, index=unigrain) # sum_labels takes care of float64 precision for big arrays with more than 16*1024**2 quaternions
        qdis_unic[:,1] = ndix.mean(qdis[:,1], grains, index=unigrain)
        qdis_unic[:,2] = ndix.mean(qdis[:,2], grains, index=unigrain)
        qdis_unic[:,3] = ndix.mean(qdis[:,3], grains, index=unigrain)

        norm = cp.sqrt(cp.einsum('...i,...i', qdis_unic, qdis_unic))
        qdis_unic /= norm[..., cp.newaxis]

        # q_mean=q_ref*q_sum/|q_sum|
        q4_mult(qref_unic, qdis_unic.astype(cp.float32), qavg_unic)

        #### theta for convergence of qavg:
        theta_unic[:] = cp.arccos( cp.minimum(cp.abs(qref_unic[:,0]*qavg_unic[:,0] +
                                                     qref_unic[:,1]*qavg_unic[:,1] +
                                                     qref_unic[:,2]*qavg_unic[:,2] +
                                                     qref_unic[:,3]*qavg_unic[:,3]), 1.), dtype=DTYPEf)*2*180/cp.pi
        theta_iter[ii] = theta_unic.max()
        qref_unic[:,:] = qavg_unic

        # free memory for next loop:
        qdis = qdis[0:1]*0.
        mempool.free_all_blocks()

        ii += 1

    # angles:
    cp.arccos(GROD, out=GROD)
    GROD *= 2*180/cp.pi

    GOS_unic = ndix.mean(GROD, grains, index=unigrain)

    theta_iter = theta_iter[theta_iter >= 0.]

    logging.info("Computed average grain orientation over {} crystals and {} grains in {} iterations.".format(len(qarr), len(unigrain), ii))
    logging.info("Theta convergence (degrees): {}".format(theta_iter))

    return qavg_unic, GOS_unic, theta_unic, GROD, theta_iter

if __name__ == "__main__":
    import doctest
    doctest.testmod()

