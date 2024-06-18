# -*- coding: utf-8 -*-

# pyorimap/quaternions_cp.py

"""
Quaternion operations using CuPy.

The module contains the following Kernels and functions:

- `q4_mult_EWK` - CuPy ElementwiseKernel for quaternion multiplication
- `q4_angle_EWK` - CuPy ElementwiseKernel to return the angle between two quaternions
- `q4_mult(qa, qb, qc)` - perform quaternion multiplication on GPU
- `q4_from_eul(eul)` - compute quaternion from Bunge Euler angles on GPU
- `q4_disori_angle(qa, qb, qc, qsym, a0` - compute the disorientation angle between `qa` and  `qb` (taking symmetries into account)
"""

import logging
import numpy as np
import cupy as cp
import quaternions_np as q4np

DTYPEf = cp.float32
DTYPEi = cp.int32

q4_mult_EWK = cp.ElementwiseKernel(
    'float32 qa0, float32 qa1, float32 qa2, float32 qa3, float32 qb0, float32 qb1, float32 qb2, float32 qb3',

    'float32 qc0, float32 qc1, float32 qc2, float32 qc3',

    '''
    qc0 = qa0*qb0 - qa1*qb1 - qa2*qb2 - qa3*qb3;
    qc1 = qa0*qb1 + qa1*qb0 + qa2*qb3 - qa3*qb2;
    qc2 = qa0*qb2 - qa1*qb3 + qa2*qb0 + qa3*qb1;
    qc3 = qa0*qb3 + qa1*qb2 - qa2*qb1 + qa3*qb0;
    ''',

    'q4_mult')

q4_angle_EWK = cp.ElementwiseKernel(
    'float32 qa0, float32 qa1, float32 qa2, float32 qa3, float32 qb0, float32 qb1, float32 qb2, float32 qb3',

    'float32 ang',

    '''
    ang = 2*acos( min(abs(qa0*qb0 + qa1*qb1 + qa2*qb2 + qa3*qb3), 1.) )*180/3.141592653589793;
    ''',

   'q4_angle')

def q4_mult(qa, qb, qc):
    """
    Quaternion multiplication, wrapper for q4_mult_EWK() using CuPy.

    `qa`, `qb`, `qc` already on GPU memory.

    Parameters
    ----------
    qa : ndarray
        (n, 4) array of quaternions or single quaternion of shape (4,)
        on GPU memory.
    qb : ndarray
        (n, 4) array of quaternions or single quaternion of shape (4,)
        on GPU memory.
    qc : ndarray
        the result of the quaternion multiplication `qa`*`qb`
        on GPU memory, modified in place.

    Examples
    --------
    >>> qa = q4np.q4_random(n=1024)
    >>> qb = q4np.q4_random(n=1024)
    >>> qc = q4np.q4_mult(qa, qb)
    >>> qa_gpu = cp.asarray(qa, dtype=DTYPEf)
    >>> qb_gpu = cp.asarray(qb, dtype=DTYPEf)
    >>> qc_gpu = cp.empty_like(qa_gpu)
    >>> q4_mult(qa_gpu, qb_gpu, qc_gpu)
    >>> np.allclose(qc, cp.asnumpy(qc_gpu), atol=1e-6)
    True
    """
    q4_mult_EWK(qa[...,0], qa[...,1], qa[...,2], qa[...,3],
                qb[...,0], qb[...,1], qb[...,2], qb[...,3],
                qc[...,0], qc[...,1], qc[...,2], qc[...,3])

def q4_from_eul(eul):
    """
    Compute quaternions from Bunge Euler angles (in degrees) on the GPU.

    CuPy version, input `eul` already on GPU memory.

    Parameters
    ----------
    eul : ndarray
        Bunge Euler angles in degrees, shape (ncrys, 3),
        on GPU memory.

    Returns
    -------
    qa : ndarray
        quaternion array of shape (ncrys, 4) and type cp.float32 by default
        on GPU memory.

    Examples
    --------
    >>> qa = q4np.q4_random(n=1024)
    >>> eul = q4np.q4_to_eul(qa)
    >>> qback = q4np.q4_from_eul(eul)
    >>> eul_gpu = cp.asarray(eul, dtype=DTYPEf)
    >>> qback_gpu = q4_from_eul(eul_gpu)
    >>> np.allclose(qa, qback, atol=1e-6)
    True
    >>> np.allclose(qback, cp.asnumpy(qback_gpu), atol=1e-6)
    True
    """
    ncrys = len(eul)

    eul *= cp.pi/180.

    qa = cp.zeros((ncrys,4), dtype=DTYPEf)
    qa[:,0] = cp.cos(eul[:,1]/2)*cp.cos((eul[:,0]+eul[:,2])/2)
    qa[:,1] = cp.sin(eul[:,1]/2)*cp.cos((eul[:,0]-eul[:,2])/2)
    qa[:,2] = cp.sin(eul[:,1]/2)*cp.sin((eul[:,0]-eul[:,2])/2)
    qa[:,3] = cp.cos(eul[:,1]/2)*cp.sin((eul[:,0]+eul[:,2])/2)

    # positive quaternion:
    whr = cp.where(qa[:,0] < 0.)
    qa[whr] *= -1

    return qa

def q4_disori_angle(qa, qb, qc, qsym, a0):
    """
    Disorientation angle (degrees) between `qa` and `qb`, taking `qsym` symmetries into account.

    CuPy version, all input arrays already on GPU memory.

    Parameters
    ----------
    qa : ndarray
        array of quaternions or single quaternion on GPU memory.
    qb : ndarray
        array of quaternions or single quaternion on GPU memory.
    qc : ndarray
        intermediate array of quaternions with the same shape as `qa`,
        modified in place on GPU memory.
    qsym : ndarray
        quaternion array of symmetry operations on GPU memory.
    a0 : ndarray
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
    >>> qc_gpu = cp.empty_like(qb_gpu)
    >>> qsym_gpu = cp.asarray(qsym, dtype=DTYPEf)
    >>> a0_gpu = cp.zeros(qa_gpu.shape[0], dtype=DTYPEf)
    >>> q4_disori_angle(qa_gpu, qb_gpu, qc_gpu, qsym_gpu, a0_gpu)
    >>> np.allclose(ang, cp.asnumpy(a0_gpu), atol=1e-3)
    True
    """

    # initialize arrays:
    a0  += 99999
    a1 = a0 * 0.
    for q in qsym:
        # update qc as the equivalent quaternion of qa:
        q4_mult_EWK(qa[...,0], qa[...,1], qa[...,2], qa[...,3],
                    q[0],      q[1],      q[2],      q[3],
                    qc[...,0], qc[...,1], qc[...,2], qc[...,3])

        # angle between qc and qb:
        q4_angle_EWK(qc[...,0], qc[...,1], qc[...,2], qc[...,3],
                     qb[...,0], qb[...,1], qb[...,2], qb[...,3],
                     a1)

        cp.minimum(a0, a1, out=a0)
