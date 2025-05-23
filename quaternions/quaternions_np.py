# -*- coding: utf-8 -*-

# pyorimap/quaternions_np.py

"""
Quaternion operations applied to (poly)crystals using regular numpy.

This module contains the following functions:

- `q4_positive(qarr)` - return positive quaternion
- `q4_sym_cubic()` - generate quaternions for cubic crystal symmetry
- `q4_sym_hex()` - generate quaternions for hexagonal crystal symmetry
- `q4_sym_tetra()` - generate quaternions for tetragonal crystal symmetry
- `q4_sym_ortho()` - generate quaternions for orthorhombic crystal symmetry
- `q4_sym_mono()` - generate quaternions for monoclinic crystal symmetry
- `q4_random()` - generate random quaternions
- `q4_from_axis_angle(axis, ang)` - generate quaternion corresponding to `ang` degrees rotation about `axis`
- `q4_mult(qa, qb)` - compute quaternion multiplication
- `q4_inv(qarr)` - compute the inverse of unit quaternions
- `q4_from_eul(eul)` - compute quaternion from Bunge Euler angles
- `q4_to_eul(qarr)` - compute Bunge Euler angles from quaternion
- `q4_from_mat(R)` - compute quaternion from rotation matrix
- `q4_to_mat(qarr)` - compute rotation matrix from quaternion
- `mat_from_eul(eul)` - compute rotation matrix from Bunge Euler angles
- `mat_to_eul(R)` - compute Bunge Euler angles from rotation matrix
- `transpose_mat(R)` - transpose the array of rotation matrices
- `q4_cosang2(qa, qb)` - cosine of the half angle between `qa` and `qb`
- `q4_disori_angle(qa, qb, qsym)` - compute the disorientation angle between `qa` and  `qb` (taking symmetries into account)
- `q4_disori_quat(qa, qb, qsym, frame)` - compute the disorientation quaternion between `qa` and  `qb` (taking symmetries into account)
- `q4_mean_disori(qarr, qsym)` - compute the average orientation and disorientation (GOS and GROD)
- `q4_orispread(ncrys, thetamax, misori)` - generate an orientation spread
"""

import logging
import numpy as np
import scipy.ndimage as ndi
import quaternion as quat

from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from pyorimap.pom_data import mtex_data as mtex

DTYPEf = np.float32
DTYPEi = np.int32
_EPS = 1e-7

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
    >>> qarr = np.array([[-1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,-1]]).astype(DTYPEf)
    >>> qpos = q4_positive(qarr)
    >>> np.allclose(qpos, -qarr, atol=1e-6)
    True
    """
    qpos = np.atleast_2d(qarr).copy()

    # where q0 < 0:
    whr1 = (qpos[:,0] < -_EPS)
    qpos[whr1] *= -1

    # where q0 close to zero (-_EPS < q0 < _EPS), check other terms:
    whr2 = ~whr1 * (qpos[:,0] < _EPS)
    if np.sum(whr2) > 0:
        qsub = qpos[whr2]
        col = (np.abs(qsub) > _EPS).argmax(axis=1)
        row = np.arange(qsub.shape[0])
        whr = (qsub[row, col] < 0)
        qsub[whr] *=  -1
        qpos[whr2] = qsub

    return np.squeeze(qpos)

def q4_sym_cubic(static=True, dtype=DTYPEf):
    """
    Compute the (24, 4) quaternion array for cubic crystal symmetry.

    Returns
    -------
    qsym : ndarray
        quaternion array of shape (24, 4) and type np.float32 by default.

    Examples
    --------
    >>> np.allclose(q4_sym_cubic(static=True), q4_sym_cubic(static=False), atol=1e-6)
    True
    >>> qsym = q4_sym_cubic()
    >>> qmtex = mtex.load_mtex_qsym(sym='cubic')
    >>> ang = np.min(2*np.arccos(np.minimum(np.abs(np.dot(qsym,qmtex.T)),1.)), axis=0)
    >>> np.allclose(ang, np.zeros(24, dtype=DTYPEf), atol=0.1)
    True
    """
    if static:
        sq2 = np.sqrt(2.)/2
        qsym = np.array([[ 1.    ,  0.    ,  0.    ,  0.    ],
                        [ 0.5   ,  0.5   ,  0.5   ,  0.5   ],
                        [-0.5   ,  0.5   ,  0.5   ,  0.5   ],
                        [-sq2,  0.    ,  0.    , -sq2],
                        [ 0.    ,  0.    , -sq2, -sq2],
                        [ sq2,  0.    , -sq2,  0.    ],
                        [ 0.    ,  0.    ,  0.    ,  1.    ],
                        [-0.5   , -0.5   ,  0.5   ,  0.5   ],
                        [-0.5   , -0.5   ,  0.5   , -0.5   ],
                        [ sq2,  0.    ,  0.    , -sq2],
                        [ sq2,  sq2,  0.    ,  0.    ],
                        [ 0.    ,  sq2,  0.    ,  sq2],
                        [ 0.    , -sq2, -sq2,  0.    ],
                        [ sq2, -sq2,  0.    ,  0.    ],
                        [ sq2,  0.    ,  sq2,  0.    ],
                        [ 0.    ,  1.    ,  0.    ,  0.    ],
                        [-0.5   ,  0.5   , -0.5   ,  0.5   ],
                        [-0.5   , -0.5   , -0.5   ,  0.5   ],
                        [ 0.    , -sq2,  sq2,  0.    ],
                        [ 0.    ,  0.    ,  sq2, -sq2],
                        [ 0.    ,  sq2,  0.    , -sq2],
                        [ 0.    ,  0.    , -1.    ,  0.    ],
                        [ 0.5   , -0.5   , -0.5   ,  0.5   ],
                        [ 0.5   , -0.5   ,  0.5   ,  0.5   ]], dtype=dtype)
    else:
        qsym = np.zeros((24,4), dtype=dtype)

        axis = np.array([[1,1,0],[0,0,1],[1,1,1]], dtype=dtype)
        ang =  np.array([ 180.,   90.,    120.  ], dtype=dtype)
        qrot = q4_from_axis_angle(axis, ang)

        q = np.array([1,0,0,0], dtype=dtype)
        isym = 0
        for l1 in range(2):
            for l2 in range(4):
                for l3 in range(3):
                    qsym[isym,:] = q
                    isym += 1
                    # ::::::::  3-fold rotation about (1 1 1)  :::::::
                    q = q4_mult(q, qrot[2])
                # :::::::::  4-fold rotation about (0 0 1)  ::::::::
                q = q4_mult(q, qrot[1])
            # ::::::::::  2-fold rotation about (1 1 0)  :::::::::
            q = q4_mult(q, qrot[0])

    return qsym

def q4_sym_hex(static=True, dtype=DTYPEf):
    """
    Compute the (12, 4) quaternion array for hexagonal crystal symmetry.

    Returns
    -------
    qsym : ndarray
        quaternion array of shape (12, 4) and type np.float32 by default.

    Examples
    --------
    >>> np.allclose(q4_sym_hex(static=True), q4_sym_hex(static=False), atol=1e-6)
    True
    >>> qsym = q4_sym_hex()
    >>> qmtex = mtex.load_mtex_qsym(sym='hex')
    >>> ang = np.min(2*np.arccos(np.minimum(np.abs(np.dot(qsym,qmtex.T)),1.)), axis=0)
    >>> np.allclose(ang, np.zeros(12, dtype=DTYPEf), atol=0.1)
    True
    """
    if static:
        sq3 = np.sqrt(3.)/2
        qsym = np.array([[ 1.   ,  0.   ,  0.   ,  0.   ],
                        [ sq3,  0.   ,  0.   ,  0.5  ],
                        [ 0.5  ,  0.   ,  0.   ,  sq3],
                        [ 0.   ,  0.   ,  0.   ,  1.   ],
                        [-0.5  ,  0.   ,  0.   ,  sq3],
                        [-sq3,  0.   ,  0.   ,  0.5  ],
                        [ 0.   , -1.   ,  0.   ,  0.   ],
                        [ 0.   , -sq3,  0.5  ,  0.   ],
                        [ 0.   , -0.5  ,  sq3,  0.   ],
                        [-0.   ,  0.   ,  1.   ,  0.   ],
                        [-0.   ,  0.5  ,  sq3,  0.   ],
                        [-0.   ,  sq3,  0.5  ,  0.   ]], dtype=dtype)
    else:
        qsym = np.zeros((12,4), dtype=dtype)

        axis = np.array([[1,0,0],[0,0,1]], dtype=dtype)
        ang =  np.array([ 180.,   60.   ], dtype=dtype)
        qrot = q4_from_axis_angle(axis, ang)

        q = np.array([1,0,0,0], dtype=dtype)
        isym = 0
        for l1 in range(2):
            for l2 in range(6):
                qsym[isym,:] = q
                isym += 1
                # ::::::::  6-fold rotation about (0 0 0 1)  :::::::
                q = q4_mult(q, qrot[1])
            # ::::::::::  2-fold rotation about (1 0 0 0)  :::::::::
            q = q4_mult(q, qrot[0])

    return qsym

def q4_sym_tetra(static=True, dtype=DTYPEf):
    """
    Compute the (8, 4) quaternion array for tetragonal crystal symmetry.

    Returns
    -------
    qsym : ndarray
        quaternion array of shape (8, 4) and type np.float32 by default.

    Examples
    --------
    >>> np.allclose(q4_sym_tetra(static=True), q4_sym_tetra(static=False), atol=1e-6)
    True
    >>> qsym = q4_sym_tetra()
    >>> qmtex = mtex.load_mtex_qsym(sym='tetra')
    >>> ang = np.min(2*np.arccos(np.minimum(np.abs(np.dot(qsym,qmtex.T)),1.)), axis=0)
    >>> np.allclose(np.degrees(ang), np.zeros(8, dtype=DTYPEf), atol=0.1)
    True
    """
    if static:
        sq2 = np.sqrt(2.)/2
        qsym = np.array([[ 1.    ,  0.    ,  0.    ,  0.    ],
                        [ sq2,  0.    ,  0.    ,  sq2],
                        [ 0.    ,  0.    ,  0.    ,  1.    ],
                        [-sq2,  0.    ,  0.    ,  sq2],
                        [ 0.    , -1.    ,  0.    ,  0.    ],
                        [ 0.    , -sq2,  sq2,  0.    ],
                        [ 0.    ,  0.    ,  1.    ,  0.    ],
                        [-0.    ,  sq2,  sq2,  0.    ]], dtype=dtype)
    else:
        qsym = np.zeros((8,4), dtype=dtype)

        axis = np.array([[1,0,0],[0,0,1]], dtype=dtype)
        ang =  np.array([ 180.,   90.   ], dtype=dtype)
        qrot = q4_from_axis_angle(axis, ang)

        q = np.array([1,0,0,0], dtype=dtype)
        isym = 0
        for l1 in range(2):
            for l2 in range(4):
                qsym[isym,:] = q
                isym += 1
                # ::::::::  4-fold rotation about (0 0 1)  :::::::
                q = q4_mult(q, qrot[1])
            # ::::::::::  2-fold rotation about (1 0 0)  :::::::::
            q = q4_mult(q, qrot[0])

    return qsym

def q4_sym_ortho(static=True, dtype=DTYPEf):
    """
    Compute the (4, 4) quaternion array for orthorhombic crystal symmetry.

    Returns
    -------
    qsym : ndarray
        quaternion array of shape (4, 4) and type np.float32 by default.

    Examples
    --------
    >>> np.allclose(q4_sym_ortho(static=True), q4_sym_ortho(static=False), atol=1e-6)
    True
    >>> qsym = q4_sym_ortho()
    >>> qmtex = mtex.load_mtex_qsym(sym='ortho')
    >>> ang = np.min(2*np.arccos(np.minimum(np.abs(np.dot(qsym,qmtex.T)),1.)), axis=0)
    >>> np.allclose(np.degrees(ang), np.zeros(4, dtype=DTYPEf), atol=0.1)
    True
    """
    if static:
        qsym = np.array([[ 1.,  0.,  0.,  0.],
                         [ 0.,  1.,  0.,  0.],
                         [ 0.,  0.,  1.,  0.],
                         [ 0.,  0.,  0.,  1.]], dtype=dtype)
    else:
        qsym = np.zeros((4,4), dtype=dtype)

        axis = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=dtype)
        ang =  np.array([ 180.,   180.,   180.  ], dtype=dtype)
        qrot = q4_from_axis_angle(axis, ang)

        q = np.array([1,0,0,0], dtype=dtype)

        qsym[0,:] = q
        qsym[1,:] = q4_mult(q, qrot[0])
        qsym[2,:] = q4_mult(q, qrot[1])
        qsym[3,:] = q4_mult(q, qrot[2])

    return qsym

def q4_sym_mono(static=True, dtype=DTYPEf):
    """
    Compute the (2, 4) quaternion array for monoclinic crystal symmetry.

    Returns
    -------
    qsym : ndarray
        quaternion array of shape (2, 4) and type np.float32 by default.

    Examples
    --------
    >>> np.allclose(q4_sym_mono(static=True), q4_sym_mono(static=False), atol=1e-6)
    True
    >>> qsym = q4_sym_mono()
    >>> qmtex = mtex.load_mtex_qsym(sym='mono')
    >>> ang = np.min(2*np.arccos(np.minimum(np.abs(np.dot(qsym,qmtex.T)),1.)), axis=0)
    >>> np.allclose(np.degrees(ang), np.zeros(2, dtype=DTYPEf), atol=0.1)
    True
    """
    if static:
        qsym = np.array([[ 1.,  0.,  0.,  0.],
                         [ 0.,  0.,  1.,  0.]], dtype=dtype)
    else:
        qsym = np.zeros((2,4), dtype=dtype)

        axis = np.array([0,1,0], dtype=dtype)
        ang =  np.array( 180., dtype=dtype)
        qrot = q4_from_axis_angle(axis, ang)

        q = np.array([1,0,0,0], dtype=dtype)

        qsym[0,:] = q
        qsym[1,:] = q4_mult(q, qrot)

    return qsym

def q4_random(n=1024, dtype=DTYPEf):
    """
    Generate a (`n`, 4) unit quaternion array of random orientations.

    Parameters
    ----------
    n : int, default=1024
        Number of orientations to generate.

    Returns
    -------
    q4 : ndarray
        quaternion array of shape (`n`, 4) and type np.float32 by default.

    Notes
    -----
    Implementation based on Christoph Gohlke's random_quaternion() in transformations.py
    at https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py.

    Examples
    --------
    >>> q = q4_random(1024)
    >>> norm = np.sqrt(np.sum(q**2, axis=1))
    >>> np.allclose(norm, np.ones(1024, dtype=DTYPEf))
    True
    """
    rand = np.random.rand(n,3).astype(dtype)

    r1 = np.sqrt(1.0 - rand[:,0])
    r2 = np.sqrt(rand[:,0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[:,1]
    t2 = pi2 * rand[:,2]

    q4 = np.zeros((n,4), dtype=dtype)
    q4[:,0] = np.cos(t2)*r2
    q4[:,1] = np.sin(t1)*r1
    q4[:,2] = np.cos(t1)*r1
    q4[:,3] = np.sin(t2)*r2

    # positive quaternion:
    q4 = q4_positive(q4)

    return np.squeeze(q4)

def q4_from_axis_angle(axis, ang, dtype=DTYPEf):
    """
    Return the quaternion corresponding to `ang` degrees rotation about `axis`.

    The number of rotations, `nrot`, is infered from the list of rotation axes
    or the list of rotations angles.

    Parameters
    ----------
    axis : array_like
        Rotation axis (or list of rotation axes).
    ang : float or array_like
        Rotation angle (or list of rotation angles).

    Returns
    -------
    qrot : ndarray
        (`nrot`, 4) quaternion array of type np.float32 by default.

    Raises
    ------
    Exception
        If the number of rotations infered from axis and ang do not match.
    ZeroDivisionError
        If one of the axis has zero norm.

    Notes
    -----
    Implementation based on Christoph Gohlke's quaternion_about_axis() in transformations.py
    at https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py.

    Examples
    --------
    >>> q = q4_from_axis_angle([1,1,1], 0)
    >>> np.allclose(q, np.array([1,0,0,0], dtype=DTYPEf))
    True
    >>> q = q4_from_axis_angle([[1,0,0],[1,1,1]], 0)
    >>> q.shape
    (2, 4)
    >>> q = q4_from_axis_angle([1,0,0], [0,1,2,3,4])
    >>> q.shape
    (5, 4)
    >>> norm = np.sqrt(np.sum(q**2, axis=1))
    >>> np.allclose(norm, np.ones(5, dtype=DTYPEf))
    True
   """

    axis = np.atleast_1d(axis).astype(dtype)
    ang = np.array(ang).astype(dtype)

    nax = 1  if axis.ndim==1 else len(axis[:,0])
    nang = 1 if ang.ndim==0 else len(ang)

    if nax == nang:
        nrot = nang
    elif (nax == 1) and (nang > 1):
        nrot = nang
    elif (nang == 1) and (nax > 1):
        nrot = nax
    else:
        logging.error("q4_from_axis_angle(): mismatch between nax and nang values")
        raise Exception(f"Mismatch between nax and nang values.")
    ang *= np.pi/180

    qrot = np.zeros((nrot, 4), dtype=dtype)
    qrot[:,1] = axis[...,0]
    qrot[:,2] = axis[...,1]
    qrot[:,3] = axis[...,2]

    #qlen = np.sqrt(np.sum(qrot**2, axis=1)) # slower
    qlen = np.sqrt(np.einsum('...i,...i', qrot, qrot))
    if qlen.min() > _EPS:
        qlen = np.sin(ang/2.0) / qlen
        qrot *= qlen[..., np.newaxis]
        #qrot[:,1] *= qlen
        #qrot[:,2] *= qlen
        #qrot[:,3] *= qlen
    else:
        logging.error("q4_from_axis_angle(): some axis vectors have a zero norm. Can't be used as to generate a quaternion.")
        raise ZeroDivisionError
    qrot[:,0] = np.cos(ang/2.0)

    # positive quaternion:
    qrot = q4_positive(qrot)

    return np.squeeze(qrot)

def q4_mult(qa, qb, dtype=DTYPEf):
    """
    Array-based quaternion multiplication.

    Parameters
    ----------
    qa : ndarray
        (n, 4) array of quaternions or single quaternion of shape (4,).
    qb : ndarray
        (n, 4) array of quaternions or single quaternion of shape (4,).

    Returns
    -------
    qc : ndarray
        the result of the quaternion multiplication `qa`*`qb`.
        Shape (n, 4) or (4,) depending on `qa` and `qb`.
        Type np.float32 by default.

    Raises
    ------
    IndexError
        If the dimensions of the input arrays do not match, taking ellipsis into account.

    Notes
    -----
    Faster CPU implementation available in https://github.com/moble/quaternion for example,
    with the implementation of a special quaternion dtype for numpy.

    Examples
    --------
    >>> qa = q4_random(n=1024)
    >>> qb = q4_random(n=1024)
    >>> qc = q4_mult(qa, qb)
    >>> qa1 = quat.as_quat_array(qa)
    >>> qb1 = quat.as_quat_array(qb)
    >>> qc1 = qa1 * qb1
    >>> np.allclose(qc, quat.as_float_array(qc1), atol=1e-6)
    True
    >>> qb = qb[0]
    >>> qc = q4_mult(qa, qb)
    >>> qb1 = qb1[0]
    >>> qc1 = qa1 * qb1
    >>> np.allclose(qc, quat.as_float_array(qc1), atol=1e-6)
    True
    """

    if (qb.ndim == 2) and (qa.ndim == 1):
        qc = np.zeros(qb.shape, dtype=dtype)
    else:
        qc = np.zeros(qa.shape, dtype=dtype)

    qc[...,0] = qa[...,0]*qb[...,0] - qa[...,1]*qb[...,1] - qa[...,2]*qb[...,2] - qa[...,3]*qb[...,3]
    qc[...,1] = qa[...,0]*qb[...,1] + qa[...,1]*qb[...,0] + qa[...,2]*qb[...,3] - qa[...,3]*qb[...,2]
    qc[...,2] = qa[...,0]*qb[...,2] - qa[...,1]*qb[...,3] + qa[...,2]*qb[...,0] + qa[...,3]*qb[...,1]
    qc[...,3] = qa[...,0]*qb[...,3] + qa[...,1]*qb[...,2] - qa[...,2]*qb[...,1] + qa[...,3]*qb[...,0]

    return qc

def q4_inv(qarr, dtype=DTYPEf):
    """
    Inverse of unit quaternion based on its conjugate.

    Parameters
    ----------
    qarr : ndarray
        array of quaternions or single quaternion.

    Returns
    -------
    qinv : ndarray
        conjugate of `qarr`.
        Type np.float32 by default.

    Examples
    --------
    >>> qa = np.array([0,1,0,0], dtype=DTYPEf)
    >>> qinv = q4_inv(qa)
    >>> np.allclose(qinv, np.array([0,-1,0,0], dtype=DTYPEf))
    True
    >>> qa = q4_random(n=1024)
    >>> qinv = q4_inv(qa)
    """

    qinv = qarr.copy()
    qinv[...,1] *=  -1
    qinv[...,2] *=  -1
    qinv[...,3] *=  -1

    return qinv

def q4_from_eul(eul, dtype=DTYPEf):
    """
    Converts Bunge Euler angles (in degrees) to quaternions.

    Parameters
    ----------
    eul : array_like
        Bunge Euler angles in degrees ['phi1', 'Phi', 'phi2'].
        If given as a structured array, 'phi1', 'Phi', 'phi2'
        must be present in eul.dtype.names.
        The number of crystals ncrys=len(eul).

    Returns
    -------
    qarr : ndarray
        quaternion array of shape (ncrys, 4) and type np.float32 by default.

    Raises
    ------
    ValueError
        If the input is a structured array and a named field
        in ['phi1', 'Phi', 'phi2'] is missing or incorrect.

    Notes
    -----
    Bunge Euler angles correspond to a "Z X Z" sequence of successive rotations from the sample frame to the crystal frame,
    where the 2nd and 3rd rotation apply on the rotated frame resulting from the previous rotations.
    The present implementation follows the conventions detailed in the documentation of the
    orilib routines by R. Quey at https://sourceforge.net/projects/orilib/.

    Examples
    --------
    >>> qarr = q4_random(n=1024)
    >>> eul = q4_to_eul(qarr)
    >>> qback = q4_from_eul(eul)
    >>> np.allclose(qarr, qback, atol=1e-6)
    True
    """
    if isinstance(eul, np.ndarray):
        if eul.dtype.names is not None: # this is a structured array, convert it to ndarray
            eul = structured_to_unstructured(eul[['phi1', 'Phi', 'phi2']])
    eul = np.atleast_2d(eul).astype(dtype)
    eul *= np.pi/180.
    ncrys = len(eul)

    phi1 = eul[:,0]
    Phi  = eul[:,1]
    phi2 = eul[:,2]

    qarr = np.zeros((ncrys,4), dtype=dtype)

    qarr[:,0] = np.cos(Phi/2)*np.cos((phi1+phi2)/2)
    qarr[:,1] = np.sin(Phi/2)*np.cos((phi1-phi2)/2)
    qarr[:,2] = np.sin(Phi/2)*np.sin((phi1-phi2)/2)
    qarr[:,3] = np.cos(Phi/2)*np.sin((phi1+phi2)/2)

    # positive quaternion:
    qarr = q4_positive(qarr)

    return np.squeeze(qarr)

def q4_to_eul(qarr, dtype=DTYPEf):
    """
    Converts quaternions to Bunge Euler angles in degrees.

    Parameters
    ----------
    qarr : array_like
        quaternion array of shape (ncrys, 4) and type np.float32 by default.
        The number of crystals ncrys=len(qarr).

    Returns
    -------
    eul : ndarray
        (ncrys, 3) array of Bunge Euler angles ('phi1', 'Phi', 'phi2') in degrees,
        where: 0 <= phi1 < 360; 0 <= Phi < 180; 0 <= phi2 < 360.

    Notes
    -----
    Bunge Euler angles correspond to a "Z X Z" sequence of successive rotations from the sample frame to the crystal frame,
    where the 2nd and 3rd rotation apply on the rotated frame resulting from the previous rotations.
    The present implementation follows the conventions detailed in the documentation of the
    orilib routines by R. Quey at https://sourceforge.net/projects/orilib/.

    Examples
    --------
    >>> eul = np.random.rand(1024,3).astype(DTYPEf)
    >>> eul[:,0] *= 360.; eul[:,1] *= 180.; eul[:,2] *= 360.
    >>> qarr = q4_from_eul(eul)
    >>> eulback = q4_to_eul(qarr)
    >>> np.allclose(eul, eulback, atol=0.2)
    True
    >>> eul = [[10,30,50],[10,0,0],[10,180,0]]
    >>> qarr = q4_from_eul(eul)
    >>> eulback = q4_to_eul(qarr)
    >>> np.allclose(eul, eulback, atol=0.2)
    True
    """
    qarr = np.atleast_2d(qarr).astype(dtype)
    ncrys = len(qarr)

    eul = np.zeros((ncrys, 3), dtype=dtype)

    x = np.sqrt( qarr[:,0]**2 + qarr[:,3]**2 )
    y = np.sqrt( qarr[:,1]**2 + qarr[:,2]**2 )
    # Phi (0 <= Phi < 180 by construction in the arctan2 function with positive y and x):
    eul[:,1] = np.degrees( 2*np.arctan2(y,x) )

    # where 0. < Phi < 180:
    whr = (eul[:,1] > 0.0001)*(eul[:,1] < 179.9999)
    eul[whr,0] = np.degrees( np.arctan2(qarr[whr,3],qarr[whr,0]) + np.arctan2(qarr[whr,2],qarr[whr,1]) )
    eul[whr,2] = np.degrees( np.arctan2(qarr[whr,3],qarr[whr,0]) - np.arctan2(qarr[whr,2],qarr[whr,1]) )

    # where Phi = 0. (or actually lower than _EPS):
    whr = (eul[:,1] < 0.0001)
    eul[whr,0] = np.degrees( 2*np.arctan2(qarr[whr,3],qarr[whr,0]) ) # to be checked against arctan2(q3,q0)
    eul[whr,1] = 0.
    eul[whr,2] = 0.

    # where Phi = 180. (or actually greater than 180.-_EPS):
    whr = (eul[:,1] > 179.9999)
    eul[whr,0] = np.degrees( 2*np.arctan2(qarr[whr,2],qarr[whr,1]) )
    eul[whr,1] = 180.
    eul[whr,2] = 0.

    eul[:,0] = eul[:,0] % 360.
    eul[:,2] = eul[:,2] % 360.

    return np.squeeze(eul)

def mat_from_eul(eul, dtype=DTYPEf):
    """
    Compute rotation matrix from Bunge Euler angles.

    Parameters
    ----------
    eul : array_like
        Bunge Euler angles in degrees ['phi1', 'Phi', 'phi2'].
        If given as a structured array, 'phi1', 'Phi', 'phi2' must be present in eul.dtype.names.
        The number of crystals ncrys=len(eul).

    Returns
    -------
    R : ndarray
        (n, 3, 3) array of rotation matrices or single rotation matrix
        of shape (3, 3) depending on input. Dtype is np.float32 by default.

    Notes
    -----
    Bunge Euler angles correspond to a "Z X Z" sequence of successive rotations from the sample frame to the crystal frame,
    where the 2nd and 3rd rotation apply on the rotated frame resulting from the previous rotations.
    Columns of the resulting rotation matrix correspond to the unit vectors
    of the sample (macroscopic) frame expressed in the crystal frame.
    The present implementation follows the conventions detailed in the documentation of the
    orilib routines by R. Quey at https://sourceforge.net/projects/orilib/.

    Examples
    --------
    >>> qarr = q4_random(n=1024)
    >>> R = q4_to_mat(qarr)
    >>> eul = mat_to_eul(R)
    >>> Rback = mat_from_eul(eul)
    >>> np.allclose(R, Rback, atol=1e-4)
    True
    """
    if isinstance(eul, np.ndarray):
        if eul.dtype.names is not None: # this is a structured array, convert it to ndarray
            eul = structured_to_unstructured(eul[['phi1', 'Phi', 'phi2']])
    eul = np.atleast_2d(eul).astype(dtype)
    eul *= np.pi/180.
    ncrys = len(eul)

    phi1 = eul[:,0]
    Phi  = eul[:,1]
    phi2 = eul[:,2]

    R = np.zeros((ncrys, 3, 3), dtype=dtype)

    R[:,0,0] =  np.cos(phi1)*np.cos(phi2) - np.sin(phi1)*np.sin(phi2)*np.cos(Phi)
    R[:,0,1] =  np.sin(phi1)*np.cos(phi2) + np.cos(phi1)*np.sin(phi2)*np.cos(Phi)
    R[:,0,2] =  np.sin(phi2)*np.sin(Phi)

    R[:,1,0] = -np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(Phi)
    R[:,1,1] = -np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(Phi)
    R[:,1,2] =  np.cos(phi2)*np.sin(Phi)

    R[:,2,0] =  np.sin(phi1)*np.sin(Phi)
    R[:,2,1] = -np.cos(phi1)*np.sin(Phi)
    R[:,2,2] =  np.cos(Phi)

    return np.squeeze(R)

def mat_to_eul(R, dtype=DTYPEf):
    """
    Compute Bunge Euler angles from rotation matrix.

    Parameters
    ----------
    R : array_like
        (n, 3, 3) array of rotation matrices or single rotation matrix
        of shape (3, 3) depending on input.

    Returns
    -------
    eul : ndarray
        (ncrys, 3) array of Bunge Euler angles ('phi1', 'Phi', 'phi2') in degrees,
        where: 0 <= phi1 < 360; 0 <= Phi < 180; 0 <= phi2 < 360.

    Notes
    -----
    Bunge Euler angles correspond to a "Z X Z" sequence of successive rotations from the sample frame to the crystal frame,
    where the 2nd and 3rd rotation apply on the rotated frame resulting from the previous rotations.
    The present implementation follows the conventions detailed in the documentation of the
    orilib routines by R. Quey at https://sourceforge.net/projects/orilib/.

    Examples
    --------
    >>> qarr = q4_random(n=1024)
    >>> eul = q4_to_eul(qarr)
    >>> mat = mat_from_eul(eul)
    >>> eulback = mat_to_eul(mat)
    >>> np.allclose(eul, eulback, atol=0.1)
    True
    """
    R = np.atleast_3d(R).reshape(-1,3,3).astype(dtype)
    ncrys = len(R)

    eul = np.zeros((ncrys, 3), dtype=dtype)

    eul[:,1] = np.degrees( np.arccos(R[:,2,2]) )
    whr = (eul[:,1] > 0.+_EPS) * (eul[:,1] < 180.-_EPS)

    eul[whr,0] = np.degrees( np.arctan2(R[whr,2,0],  -R[whr,2,1]) ) % 360.
    eul[whr,2] = np.degrees( np.arctan2(R[whr,0,2],   R[whr,1,2]) ) % 360.

    eul[~whr,0] = np.degrees( np.arctan2(R[~whr,0,1], R[~whr,0,0]) ) % 360.
    eul[~whr,2] = 0.

    return np.squeeze(eul)

def q4_from_mat(R, dtype=DTYPEf):
    """
    Compute crystal quaternion from rotation matrix.

    This implementation is indirect, transiting through Euler angles.

    Parameters
    ----------
    R : array_like
        (n, 3, 3) array of rotation matrices or single rotation matrix
        of shape (3, 3) depending on input.

    Returns
    -------
    qarr : ndarray
        quaternion array of shape (ncrys, 4) and type np.float32 by default.

    Examples
    --------
    >>> qarr = q4_random(n=1024)
    >>> R = q4_to_mat(qarr)
    >>> qback = q4_from_mat(R)
    >>> np.allclose(qarr, qback, atol=1e-4)
    True
    """
    qarr = q4_from_eul( mat_to_eul(R) )
    return qarr

def q4_to_mat(qarr, dtype=DTYPEf):
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
    >>> R = mat_from_eul([[0,45,30],[10,20,30],[0,180,10]])
    >>> qarr = q4_from_mat(R)
    >>> Rback = q4_to_mat(qarr)
    >>> np.allclose(R, Rback, atol=1e-6)
    True
    """
    qarr = np.atleast_2d(qarr)
    ncrys = len(qarr)
    R = np.zeros((ncrys, 3, 3), dtype=dtype)

    q0 = qarr[:,0]; q1 = qarr[:,1]; q2 = qarr[:,2]; q3 = qarr[:,3]

    R[:,0,0] = q0**2+q1**2-1./2
    R[:,1,0] = q1*q2-q0*q3
    R[:,2,0] = q1*q3+q0*q2
    R[:,0,1] = q1*q2+q0*q3
    R[:,1,1] = q0**2+q2**2-1./2
    R[:,2,1] = q2*q3-q0*q1
    R[:,0,2] = q1*q3-q0*q2
    R[:,1,2] = q2*q3+q0*q1
    R[:,2,2] = q0**2+q3**2-1./2

    R *= 2.

    return np.squeeze(R)

def transpose_mat(R):
    """
    Transpose the array of rotation matrices.

    Parameters
    ----------
    R : ndarray
        (n, 3, 3) array of rotation matrices or single rotation matrix
        of shape (3, 3) depending on input. Dtype is np.float32 by default.

    Returns
    -------
    Rt : ndarray
        (n, 3, 3) array of the transposed rotation matrices or
        single transposed matrix of shape (3, 3) depending on input.
        Dtype is np.float32 by default.

    Examples
    --------
    >>> qarr = q4_random(n=1024)
    >>> Rsa2cr = q4_to_mat(qarr)
    >>> Rcr2sa = transpose_mat(Rsa2cr)
    >>> Rback = transpose_mat(Rcr2sa)
    >>> np.allclose(Rsa2cr, Rback)
    True
    """

    Rt = np.einsum('...ij->...ji', R)

    return Rt

def q4_angle(qa, qb, dtype=DTYPEf):
    """
    Returns the angle (degrees) between quaternions `qa` and `qb`.

    Parameters
    ----------
    qa : ndarray
        array of quaternions or single quaternion.
    qb : ndarray
        array of quaternions or single quaternion.

    Returns
    -------
    ang : ndarray
        the angle in degrees between quaternions `qa` and `qb`.
    """
    ang = np.arccos(q4_cosang2(qa, qb))*2*180/np.pi
    return ang

def q4_cosang2(qa, qb, dtype=DTYPEf):
    """
    Returns the cosine of the half angle between quaternions `qa` and `qb`.

    Parameters
    ----------
    qa : ndarray
        array of quaternions or single quaternion.
    qb : ndarray
        array of quaternions or single quaternion.

    Returns
    -------
    ang : ndarray
        the cosine of the half angle between quaternions `qa` and `qb`.

    Raises
    ------
    IndexError
        If the dimensions of the input arrays do not match, taking ellipsis into account.

    Examples
    --------
    >>> qa = q4_random(1024)
    >>> ang = np.random.rand(1024)*180.
    >>> qrot = q4_from_axis_angle(np.random.rand(1024,3), ang)
    >>> qb = q4_mult(qa, qrot)
    >>> cosa2 = q4_cosang2(qa, qb)
    >>> aback = np.degrees(2*np.arccos(cosa2))
    >>> np.allclose(aback, ang, atol=0.1)
    True
    """

    #ang = np.minimum(np.abs(np.sum(qa*qb, axis=1)), 1.) # slower
    ang = np.minimum(np.abs(qa[...,0]*qb[...,0] +
                            qa[...,1]*qb[...,1] +
                            qa[...,2]*qb[...,2] +
                            qa[...,3]*qb[...,3]), 1.)
    return ang

def q4_disori_angle(qa, qb, qsym, method=1, return_index=False, dtype=DTYPEf):
    """
    Disorientation angle (degrees) between `qa` and `qb`, taking `qsym` symmetries into account.

    Parameters
    ----------
    qa : ndarray
        array of quaternions or single quaternion.
    qb : ndarray
        array of quaternions or single quaternion.
    qsym : ndarray
        quaternion array of symmetry operations.
    method : int, default=1
        the method to compute disorientation: 1 or 2.
        Method 1 is faster.
    return_index : bool, default=False
        whether to return also the index of the i_th equivalent quaternion yielding minimum disorientation.

    Returns
    -------
    ang : ndarray
        the minumum angle (in degrees) between quaternions `qa` and `qb`, taking symmetries into account.
    ii : ndarray
        if `return_index`=True, the index of the i_th equivalent quaternion corresponding to the minimum disorientation.

    Raises
    ------
    IndexError
        If the dimensions of the input arrays do not match, taking ellipsis into account.

    Examples
    --------
    >>> qa = q4_random(1)
    >>> qsym = q4_sym_cubic()
    >>> qequ = q4_mult(qa, qsym)
    >>> ang = q4_disori_angle(qequ, qequ[::-1,:], qsym)
    >>> np.allclose(ang, np.zeros(24, dtype=DTYPEf), atol=0.1)
    True
    >>> qa = q4_random(1024)
    >>> qb = q4_random(1024)
    >>> ang1, ii1 = q4_disori_angle(qa, qb, qsym, method=1, return_index=True)
    >>> ang2, ii2 = q4_disori_angle(qa, qb, qsym, method=2, return_index=True)
    >>> np.allclose(ang1, ang2, atol=0.1)
    True
    >>> np.allclose(ii1, ii2)
    True
    """

    if qb.ndim == 1:
        ang = np.zeros(np.atleast_2d(qa).shape[0], dtype=dtype)
    else:
        ang = np.zeros(np.atleast_2d(qb).shape[0], dtype=dtype)
    if return_index:
        ii = np.zeros(ang.shape, dtype=DTYPEi)

    if method == 1:
        # disorientation qc between qa and qb expressed in the frame of crystal a:
        qc = q4_mult(q4_inv(qa), qb)
        for isym, q in enumerate(qsym):
            # cosine of the half angle between qc and the ith symmetry operator in qsym:
            ang1 = np.atleast_1d( q4_cosang2(qc, q) )
            if return_index:
                whr = (ang1 > ang)
                ang[whr] = ang1[whr]
                ii[whr] = isym
            else:
                ang = np.maximum(ang, ang1)
    else:
        for isym, q in enumerate(qsym):
            qc = q4_mult(qa, q)
            ang1 = np.atleast_1d( q4_cosang2(qc, qb) )
            if return_index:
                whr = (ang1 > ang)
                ang[whr] = ang1[whr]
                ii[whr] = isym
            else:
                ang = np.maximum(ang, ang1)
    if return_index:
        return np.degrees(2*np.arccos(ang)), ii
    else:
        return np.degrees(2*np.arccos(ang))

def q4_disori_quat(qa, qb, qsym, frame='ref', method=1, return_index=False, dtype=DTYPEf):
    """
    Disorientation quaternion between `qa` and `qb`, taking `qsym` symmetries into account.

    Parameters
    ----------
    qa : ndarray
        array of quaternions or single quaternion.
    qb : ndarray
        array of quaternions or single quaternion.
    qsym : ndarray
        quaternion array of symmetry operations.
    frame : str, default='ref'
        the frame to express the disorientation quaternion, 'ref' or 'crys_a'.
    method : int, default=1
        the method to compute disorientation: 1 or 2.
        Method 1 is faster.
    return_index : bool, default=False
        whether to return also the index of the i_th equivalent quaternion yielding minimum disorientation.

    Returns
    -------
    qdis : ndarray
        quaternion representing the minimum disorientation from `qa` to `qb`, taking `qsym` symmetries into account.
    ii : ndarray
        if `return_index`=True, the index of the i_th equivalent quaternion corresponding to the minimum disorientation.

    Examples
    --------
    >>> qa = np.array([1,0,0,0], dtype=DTYPEf)
    >>> qsym = q4_sym_cubic(); isym = 5
    >>> qb = q4_mult(qa, qsym[isym])
    >>> qdis_ref = q4_disori_quat(qa, qb, qsym, frame='ref')
    >>> qdis_cra = q4_disori_quat(qa, qb, qsym, frame='crys_a')
    >>> np.allclose(qdis_ref, qdis_cra, atol=1e-6)
    True
    >>> qa = q4_random(1024)
    >>> qb = q4_random(1024)
    >>> ang = q4_disori_angle(qa, qb, qsym)
    >>> qdis1 = q4_disori_quat(qa, qb, qsym, frame='ref', method=1)
    >>> qdis2 = q4_disori_quat(qa, qb, qsym, frame='ref', method=2)
    >>> np.allclose(qdis1, qdis2, atol=1e-6)
    True
    >>> ang1 = np.arccos(np.abs(qdis1[:,0]))*2*180/np.pi
    >>> ang2 = np.arccos(np.abs(qdis2[:,0]))*2*180/np.pi
    >>> np.allclose(ang, ang1, atol=0.1)
    True
    >>> np.allclose(ang, ang2, atol=0.1)
    True
    >>> qdis1 = q4_disori_quat(qa, qb, qsym, frame='crys_a', method=1)
    >>> qdis2 = q4_disori_quat(qa, qb, qsym, frame='crys_a', method=2)
    >>> np.allclose(qdis1, qdis2, atol=1e-6)
    True
    """
    if method == 1:
        ang, ii = q4_disori_angle(qa, qb, qsym, method=1, return_index=True)
        qa_inv = q4_inv(q4_mult(qa, qsym[ii]))
        if frame == 'ref':
            qdis = q4_mult(qb, qa_inv)
        else:
            qdis = q4_mult(qa_inv, qb)
    else:
        if qb.ndim == 1:
            qdis = np.zeros(qa.shape, dtype=dtype)
        else:
            qdis = np.zeros(qb.shape, dtype=dtype)
        if return_index:
            ii = np.zeros(np.atleast_2d(qdis).shape[0], dtype=np.uint8)

        #qa_inv = q4_inv(qa)
        for isym, q in enumerate(qsym):
            #qb_equ = q4_mult(qb, q)
            qa_inv = q4_inv(q4_mult(qa, q))
            if frame == 'ref':
                #qtmp = q4_mult(qb_equ, qa_inv)
                qtmp = q4_mult(qb, qa_inv)
            else:
                #qtmp = q4_mult(qa_inv, qb_equ)
                qtmp = q4_mult(qa_inv, qb)
            a0 = np.minimum(np.abs(qdis[...,0]), 1.)
            a1 = np.minimum(np.abs(qtmp[...,0]), 1.)
            whr = (a1 > a0)
            qdis[whr] = qtmp[whr]
            if return_index:
                ii[whr] = isym

    qdis = q4_positive(qdis)

    if return_index:
        return qdis, ii
    else:
        return qdis

def q4_to_FZ(qarr, qsym, return_index=False, dtype=DTYPEf):
    """
    Move quaternions to Fundamental Zone based on crystal symmetries.

    The Fundamental Zone corresponds to the i_th equivalent orientation with the lowest angle wrt the reference frame.

    Parameters
    ----------
    qarr : ndarray
        array of quaternions or single quaternion.
    qsym : ndarray
        quaternion array of symmetry operations.
    return_index : bool, default=False
        whether to return also the index of the i_th equivalent quaternion corresponding to Fundamental Zone.

    Returns
    -------
    qFZ : ndarray
        equivalent quaternion array in the Fundamental Zone.
    ii : ndarray
        if `return_index`=True, the index of the i_th equivalent quaternion corresponding to Fundamental Zone.

    Examples
    --------
    >>> qa = q4_sym_cubic()
    >>> qFZ = q4_to_FZ(qa, qa)
    >>> ref = np.tile(np.array([1,0,0,0], dtype=DTYPEf), 24).reshape(24, -1)
    >>> np.allclose(qFZ, ref, atol=1e-6)
    True
    """
    a0 = 0.
    qFZ = np.zeros_like(qarr)
    if return_index:
        ii = np.zeros(np.atleast_2d(qarr).shape[0], dtype=np.uint8)

    for isym, q in enumerate(qsym):
        qequ = q4_mult(qarr, q)
        a0 = np.minimum(np.abs( qFZ[...,0]), 1.)
        a1 = np.minimum(np.abs(qequ[...,0]), 1.)
        whr = (a1 > a0)
        qFZ[whr] = qequ[whr]
        if return_index:
            ii[whr] = isym

    if return_index:
        return q4_positive(qFZ), ii
    else:
        return q4_positive(qFZ)

def q4_mean_disori(qarr, qsym, dtype=DTYPEf):
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
    GROD_stat : array_like
        [mean, std, min, Q1, median, Q3, max] of the grain reference orientation deviation (GROD).
        GROD_stat[0] is the grain orientation spread (GOS), i.e. the average disorientation angle in degrees.
    theta_iter : ndarray
        convergence angle (degree) during the iterations for `qavg`.

    Examples
    --------
    >>> qa = q4_random(1)
    >>> qarr = q4_mult(qa, q4_orispread(ncrys=1024, thetamax=2., misori=True))
    >>> qsym = q4_sym_cubic()
    >>> qavg, GROD, GROD_stat, theta_iter = q4_mean_disori(qarr, qsym)
    >>> ang = q4_angle(qa, qavg)
    >>> np.allclose(ang, 0., atol=0.5)
    True
    """

    ii = 0
    theta = 999.
    mxtheta = 0.2
    nitermax = 10
    theta_iter = np.zeros(nitermax, dtype=dtype) - 1.

    # initialize avg orientation:
    #qref = qarr[0,:]
    qmed = np.atleast_2d(np.median(qarr, axis=0))
    cosang = np.minimum(np.abs( qmed[0,0]*qarr[:,0] +
                                qmed[0,1]*qarr[:,1] +
                                qmed[0,2]*qarr[:,2] +
                                qmed[0,3]*qarr[:,3]), 1.)
    imed = np.argmax(cosang)
    qref = qarr[imed,:]
    while (theta > mxtheta) and (ii < nitermax):
        # disorientation of each crystal wrt average orientation:
        qdis = q4_disori_quat(qref, qarr, qsym, frame='crys_a', method=1)

        #qtmp = np.sum(qdis, axis=0) # careful with np.float32 sum of very big arrays with more than 16*1024**2 quaternions
        #qtmp /= np.sqrt(np.einsum('...i,...i', qtmp, qtmp))
        #qtmp /= np.sqrt(np.sum(qtmp**2)) # slower ?
        qtmp = np.mean(qdis, axis=0, dtype=np.float64)
        qtmp /= np.sqrt(qtmp[0]**2 + qtmp[1]**2 + qtmp[2]**2 + qtmp[3]**2)
        qtmp = qtmp.astype(dtype)

        # q_mean=q_ref*q_sum/|q_sum|
        qavg = q4_mult(qref, qtmp)

        #### theta for convergence of qavg:
        theta = np.arccos(q4_cosang2(qref, qavg))*2*180/np.pi
        theta_iter[ii] = theta
        qref = qavg

        ii += 1

    # angles:
    GROD = np.minimum(np.abs(qdis[:,0]), 1.)
    GROD = np.arccos(GROD)*2*180/np.pi

    GOS = GROD.mean()
    GROD_stat = [GOS,
                 GROD.std(),
                 GROD.min(),
                 np.quantile(GROD, 0.25),
                 np.median(GROD),
                 np.quantile(GROD, 0.75),
                 GROD.max()]

    theta_iter = theta_iter[theta_iter >= 0.]

    #logging.info("Computed average grain orientation over {} crystals in {} iterations.".format(len(qarr), ii))
    #logging.info("Theta convergence (degrees): {}".format(theta_iter))

    return qavg, GROD, np.array(GROD_stat), theta_iter

def q4_mean_multigrain(qarr, qsym, unigrain, iunic, iback, dtype=DTYPEf):
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
    >>> qa = q4_random(256)
    >>> grains = np.repeat(np.arange(0,256), 1024) + 1
    >>> np.random.shuffle(grains)
    >>> unic, iunic, iback = np.unique(grains, return_index=True, return_inverse=True)
    >>> qarr = q4_mult(qa[grains - 1], q4_orispread(ncrys=1024*256, thetamax=2., misori=True))
    >>> qsym = q4_sym_cubic()
    >>> qavg, GOS, theta, GROD, theta_iter = q4_mean_multigrain(qarr, qsym, unic, iunic, iback)
    >>> ang = q4_angle(qa, qavg)
    >>> np.allclose(ang, np.zeros_like(ang), atol=0.5)
    True
    """

    ii = 0
    mxtheta = 0.2
    nitermax = 3
    theta_iter = np.zeros(nitermax, dtype=dtype) - 1.

    grains = unigrain[iback]

    #unic, iunic, iback, counts = np.unique(grains, return_index=True, return_inverse=True, return_counts=True)
    theta_unic = np.zeros(len(unigrain), dtype=dtype) + 999.
    theta = theta_unic[iback]
    qdis = np.zeros_like(qarr)

    # update iunic to account for the median quaternion in each grain, instead of the first, to initialize the average loop:
    qmed = np.zeros((len(unigrain), 4), dtype=dtype)
    qmed[:,0] = ndi.median(qarr[:,0], grains, index=unigrain)
    qmed[:,1] = ndi.median(qarr[:,1], grains, index=unigrain)
    qmed[:,2] = ndi.median(qarr[:,2], grains, index=unigrain)
    qmed[:,3] = ndi.median(qarr[:,3], grains, index=unigrain)
    qmed = qmed[iback] # back to full size ncrys
    cosang = q4_cosang2(qmed, qarr)

    imed = ndi.maximum_position(cosang, grains, index=unigrain)
    imed = np.squeeze(np.array(imed, dtype=DTYPEi))
    qref_unic = qarr[np.atleast_1d(imed)]
    #qref_unic = qarr[iunic]

    while (theta_unic.max() > mxtheta) and (ii < nitermax):
        qref_tot = qref_unic[iback]

        # disorientation of each crystal wrt average orientation:
        whrT = (theta > mxtheta)
        qdis[whrT] = q4_disori_quat(qref_tot[whrT], qarr[whrT], qsym, frame='crys_a', method=1, return_index=False)
        #qdis = q4_disori_quat(qref_tot, qarr, qsym, frame='crys_a', method=1, return_index=False)

        qdis_unic = np.zeros(qref_unic.shape, dtype=np.float64)
        qdis_unic[:,0] = ndi.mean(qdis[:,0], grains, index=unigrain) # careful with np.float32 sum of very big arrays with more than 16*1024**2 quaternions
        qdis_unic[:,1] = ndi.mean(qdis[:,1], grains, index=unigrain)
        qdis_unic[:,2] = ndi.mean(qdis[:,2], grains, index=unigrain)
        qdis_unic[:,3] = ndi.mean(qdis[:,3], grains, index=unigrain)

        norm = np.sqrt(np.einsum('...i,...i', qdis_unic, qdis_unic))
        qdis_unic /= norm[..., np.newaxis]
        qdis_unic = qdis_unic.astype(dtype)

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

def q4_orispread(ncrys=1024, thetamax=1., misori=True, dtype=DTYPEf):
    """
    Generate an orientation spread.

    Implementation derived from orilib routines by R. Quey at <https://sourceforge.net/projects/orilib/>.

    Parameters
    ----------
    ncrys : int, default=1024
        number of crystals in the distribution to be generated.
    thetamax : float, default=1.0
        maximum disorientation angle in the distribution.
    misori : bool, default=True
        whether to compute an orientation spread or a misorientation spread.
        The distribution of the misorientation angles is uniform only if misori=True.

    Returns
    -------
    qarr : ndarray
        (`ncrys`, 4) array of quaternions in the distribution.

    Examples
    --------
    >>> qarr = q4_orispread(ncrys=1024, thetamax=2., misori=True)
    >>> qsym = q4_sym_cubic()
    >>> qavg, GROD, GROD_stat, theta_iter = q4_mean_disori(qarr, qsym)
    >>> np.allclose(qavg, np.array([1,0,0,0], dtype=DTYPEf), atol=1e-3)
    True
    >>> np.allclose(1., GROD.mean(), atol=0.1)
    True
    """
    rand = np.random.rand(ncrys, 3).astype(dtype)
    qarr = np.zeros((ncrys, 4), dtype=dtype)

    alpha = np.arccos (2. * rand[:,0] - 1)
    beta  = 2. * np.pi * rand[:,1]

    # temporarily storing the axis vector:
    qarr[:,1] = np.sin(alpha) * np.cos(beta)
    qarr[:,2] = np.sin(alpha) * np.sin(beta)
    qarr[:,3] = np.cos(alpha)

    thetamax = thetamax % 180.
    thetamax *= np.pi / 180.
    if misori:
        theta = thetamax * rand[:,2]
    else:
        theta = thetamax * rand[:,2]**(1./3.)

    qarr[:,0] = np.cos(theta/2.)
    qarr[:,1] *= np.sin(theta/2.)
    qarr[:,2] *= np.sin(theta/2.)
    qarr[:,3] *= np.sin(theta/2.)

    return qarr

def spherical_proj(vec, proj="stereo", north=3, angles="deg", dtype=DTYPEf):
    """
    Performs stereographic or equal-area projection of vector `vec` in the equatorial plane.

    Parameters
    ----------
    vec : ndarray
        (ncrys, 3) array of unit vectors to be projected.
    proj : str, default='stereo'
        type of projection, 'stereo' for stereographic or 'equal-area'.
    north : int, default=3
        North pole defining the projection plane.
    angles : str, default='deg'
        unit to return polar angles, degrees or radians.

    Returns
    -------
    xyproj : ndarray
        (ncrys, 2) array of projected coordinates in the equatorial plane.
    albeta : ndarray
        (ncrys, 2) array of [alpha, beta] polar angles in degrees or radians depending on `angles` parameter.
    reverse : ndarray
        boolean array indicating where the input unit vectors were pointing to the Southern hemisphere and reversed.

    Examples
    --------
    >>> vec = np.random.rand(1024,3).astype(DTYPEf)
    >>> norm = np.sqrt(np.sum(vec**2, axis=1))
    >>> vec /= norm[..., np.newaxis]
    >>> xyproj0, albeta0, reverse0 = spherical_proj(vec, proj="stereo", north=3)
    >>> xyproj1, albeta1, reverse1 = spherical_proj(vec, proj="equal-area", north=3)
    >>> np.allclose(albeta0, albeta1, atol=0.5)
    True
    """
    pi2 = 2.*np.pi
    rad2deg = 360./pi2
    if north == 1:
        x1 = 1; x2 = 2; x3 = 0
    elif north == 2:
        x1 = 2; x2 = 0; x3 = 1
    elif north == 3:
        x1 = 0; x2 = 1; x3 = 2
    else:
        x1 = 0; x2 = 1; x3 = 2

    vec = np.atleast_2d(vec)
    albeta = np.zeros((len(vec),2), dtype=dtype)
    xyproj = np.zeros((len(vec),2), dtype=dtype)

    # check Northern hemisphere:
    reverse = (vec[:,x3] < 0.)
    vec[reverse] *= -1

    # alpha:
    vec[:,x3] = np.minimum(vec[:,x3],1.)
    vec[:,x3] = np.maximum(vec[:,x3],-1.)
    albeta[:,0] = np.arccos(vec[:,x3])
    # beta:
    whr = (np.abs(albeta[:,0]) > _EPS)
    tmp = vec[:,x1][whr]/np.sin(albeta[:,0][whr])
    tmp = np.minimum(tmp,1.)
    tmp = np.maximum(tmp,-1.)
    albeta[:,1][whr] = np.arccos(tmp)
    albeta[:,1][vec[:,x2] < 0.] *= -1
    albeta[:,1] = albeta[:,1] % pi2

    xyproj[:,0] = np.cos(albeta[:,1])
    xyproj[:,1] = np.sin(albeta[:,1])
    if proj == "stereo": # stereographic projection
        Op = np.tan(albeta[:,0]/2.)
    else:                # equal-area projection
        Op = np.sin(albeta[:,0]/2.)*np.sqrt(2.)
    xyproj *= Op[..., np.newaxis]

    if angles == 'deg':
        albeta *= rad2deg

    return xyproj, albeta, reverse

def max_albeta_fundamental(albeta, sym="cubic", dtype=DTYPEf):
    """
    Maximum alpha, beta values for a given crystal symmetry in the fundamental sector.

    Parameters
    ----------
    albeta : ndarray
        (ncrys, 2) array of [alpha, beta] polar angles in radians.
    sym : str, default = 'cubic'
        crystal symmetry

    Returns
    -------
    alpha_max : ndarray
        (ncrys,) array of maximum alpha (radians) in the fundamental sector.
    beta_max : ndarray
        (ncrys,) array of maximum beta (radians) in the fundamental sector.
    """
    ncrys = len(np.atleast_2d(albeta))
    if sym == 'cubic':
        beta = albeta[:,1]
        if beta.max() > np.pi/4:
            beta =  beta % (np.pi/2)
            whr = (beta > np.pi/4)
            beta[whr] = np.pi/2 - beta[whr]
        alpha_max = np.arccos(np.sqrt(1./(2. + np.tan(beta)**2)), dtype=dtype)
        beta_max =  np.ones(ncrys, dtype=dtype)*np.pi/4
    elif sym == 'hex':
        alpha_max = np.ones(ncrys, dtype=dtype)*np.pi/2
        beta_max =  np.ones(ncrys, dtype=dtype)*np.pi/6
    elif sym == 'tetra':
        alpha_max = np.ones(ncrys, dtype=dtype)*np.pi/2
        beta_max =  np.ones(ncrys, dtype=dtype)*np.pi/4
    elif sym == 'ortho':
        alpha_max = np.ones(ncrys, dtype=dtype)*np.pi/2
        beta_max =  np.ones(ncrys, dtype=dtype)*np.pi/2
    elif sym == 'mono':
        alpha_max = np.ones(ncrys, dtype=dtype)*np.pi/2
        beta_max =  np.ones(ncrys, dtype=dtype)*np.pi
    else:
        logging.error("Z!! undefined symmetry {} for IPF projection.".format(sym))
        alpha_max = np.ones(ncrys, dtype=dtype)*np.pi/2
        beta_max =  np.ones(ncrys, dtype=dtype)*np.pi*2

    return alpha_max, beta_max

def q4_to_IPF(qarr, axis=[1,0,0], qsym=q4_sym_cubic(), proj="stereo", north=3, method=1, dtype=DTYPEf):
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
    proj : str, default='stereo'
        type of projection, 'stereo' for stereographic or 'equal-area'.
    north : int, default=3
        North pole defining the projection plane.
    method : int, default=1
        method to bring the sample vector in the elementary subspace (method 1 is faster).

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
    >>> qarr = q4_random(1024)
    >>> qsym = q4_sym_cubic()
    >>> axis = np.array([1,0,0], dtype=DTYPEf)
    >>> xyproj1, RGB1, albeta1, isym1 = q4_to_IPF(qarr, axis, qsym, proj="stereo", north=3, method=1)
    >>> xyproj2, RGB2, albeta2, isym2 = q4_to_IPF(qarr, axis, qsym, proj="stereo", north=3, method=2)
    >>> np.allclose(xyproj1, xyproj2, atol=1e-3)
    True
    >>> np.allclose(RGB1, RGB2, atol=1e-3)
    True
    >>> np.allclose(albeta1, albeta2, atol=0.5)
    True
    """
    if north == 1:
        x1 = 1; x2 = 2; x3 = 0
    elif north == 2:
        x1 = 2; x2 = 0; x3 = 1
    elif north == 3:
        x1 = 0; x2 = 1; x3 = 2
    else:
        logging.warning("Z!! choice of north pole: {} (should be 1 for [100], 2 for [010], 3 for [001])".format(north))
        x1 = 0; x2 = 1; x3 = 2

    deg2rad = np.pi/180.
    norm = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2 )
    axis = np.atleast_1d(axis) / norm
    ncrys = len(qarr)
    nsym = len(qsym)
    if nsym == 24:
        sym = 'cubic' if np.allclose(qsym, q4_sym_cubic(), atol=1e-6) else 'NA'
    elif nsym == 12:
        sym = 'hex'   if np.allclose(qsym, q4_sym_hex(), atol=1e-6) else 'NA'
    elif nsym == 8:
        sym = 'tetra' if np.allclose(qsym, q4_sym_tetra(), atol=1e-6) else 'NA'
    elif nsym == 4:
        sym = 'ortho' if np.allclose(qsym, q4_sym_ortho(), atol=1e-6) else 'NA'
    elif nsym == 2:
        sym = 'mono'  if np.allclose(qsym, q4_sym_mono(), atol=1e-6) else 'NA'
    else:
        sym = 'unknown'

    albeta = np.zeros((ncrys,2), dtype=dtype) + 360
    xyproj = np.zeros((ncrys,2), dtype=dtype)
    isym   = np.zeros(ncrys, dtype=np.uint8)

    if method == 1:
        Rsa2cr = q4_to_mat(qarr)
        #vec = np.dot(Rsa2cr, axis).astype(dtype)
        vec = np.matvec(Rsa2cr, axis, dtype=dtype)
        vec = np.atleast_2d(vec)
        #### bring projection into the elementary subspace for the RGB color code:
        if sym == 'cubic':
            #### ensure vec(2)>=vec(0)>=vec(1)>=0 in the elementary subspace:
            tmp = np.sort(np.abs(vec), axis=1)
            vec[:,x1] = tmp[:,1]
            vec[:,x2] = tmp[:,0]
            vec[:,x3] = tmp[:,2]

            xyproj, albeta, reverse = spherical_proj(vec, proj=proj, north=north, angles='rad', dtype=dtype)
        else:
            if sym == 'hex':
                xyproj1, albeta1, reverse = spherical_proj(vec, proj=proj, north=north, angles='rad', dtype=dtype)
                alpha = albeta1[:,0]
                beta =  albeta1[:,1] % (np.pi/3)
                whr = (beta > np.pi/6)
                beta[whr] = np.pi/3 - beta[whr]
            elif sym == 'tetra':
                xyproj1, albeta1, reverse = spherical_proj(vec, proj=proj, north=north, angles='rad', dtype=dtype)
                alpha = albeta1[:,0]
                beta =  albeta1[:,1] % (np.pi/2)
                whr = (beta > np.pi/4)
                beta[whr] = np.pi/2 - beta[whr]
            elif sym == 'ortho':
                xyproj1, albeta1, reverse = spherical_proj(vec, proj=proj, north=north, angles='rad', dtype=dtype)
                alpha = albeta1[:,0]
                beta =  albeta1[:,1] % (np.pi)
                whr = (beta > np.pi/2)
                beta[whr] = np.pi - beta[whr]
            elif sym == 'mono':
                xyproj1, albeta1, reverse = spherical_proj(vec, proj=proj, north=north, angles='rad', dtype=dtype)
                alpha = albeta1[:,0]
                beta =  albeta1[:,1]
                whr = (beta > np.pi)
                beta[whr] = 2*np.pi - beta[whr]

            xyproj[:,0] = np.cos(beta)
            xyproj[:,1] = np.sin(beta)
            if proj == "stereo": # stereographic projection
                Op = np.tan(alpha/2.)
            else:                # equal-area projection
                Op = np.sin(alpha/2.)*np.sqrt(2.)
            xyproj *= Op[..., np.newaxis]
            albeta[:,0] = alpha
            albeta[:,1] = beta
        alpha_max, beta_max = max_albeta_fundamental(albeta, sym)
    else:
        for iq, q in enumerate(qsym):
            qequ = q4_mult(qarr, q)
            Rsa2cr = q4_to_mat(qequ)
            vec = np.matvec(Rsa2cr, axis, dtype=dtype)
            #vec = np.dot(Rsa2cr, axis).astype(dtype)

            xyproj1, albeta1, reverse = spherical_proj(vec, proj=proj, north=north, angles='rad', dtype=dtype)

            _EPS = np.arccos(1. - 1./2**24, dtype=dtype) # float32 arccos precision
            if sym == 'cubic':
                alpha_max, beta_max = max_albeta_fundamental(albeta1, sym)
                whr = (albeta1[:,1]-_EPS < np.pi/4) * (albeta1[:,0]-_EPS < alpha_max)
            elif sym == 'hex':
                whr = (albeta1[:,1]-_EPS < np.pi/6)
            elif sym == 'tetra':
                whr = (albeta1[:,1]-_EPS < np.pi/4)
            elif sym == 'ortho':
                whr = (albeta1[:,1]-_EPS < np.pi/2)
            elif sym == 'mono':
                whr = (albeta1[:,1]-_EPS < np.pi)
            else:
                whr = (albeta1[:,1] > -_EPS)

            xyproj[whr,:] = xyproj1[whr,:]
            albeta[whr,:] = albeta1[whr,:]
            isym[whr] = iq
        alpha_max, beta_max = max_albeta_fundamental(albeta, sym)

    RGB = np.zeros((ncrys,3), dtype=dtype)
    RGB[:,0] =  1. - albeta[:,0]/alpha_max
    RGB[:,1] = (1. - albeta[:,1]/beta_max) * albeta[:,0]/alpha_max
    RGB[:,2] = (albeta[:,1]/beta_max)      * albeta[:,0]/alpha_max
    mx = RGB.max(axis=1)
    RGB /= mx[..., np.newaxis]
    #RGB = np.uint8( np.round(RGB*255/ mx[..., np.newaxis], decimals=0) )

    logging.info("Computed IPF projection for axis {}.".format(axis))

    return xyproj, RGB, np.degrees(albeta), isym

#def XXXq4_from_mat(R, dtype=DTYPEf):
#    """
#    Converts rotation matrices to quaternions.
#    Z!! ... not correct at this point ...
#    """
#    R = np.atleast_3d(R).reshape(-1,3,3)
#    ncrys = len(R)
#
#    qarr = np.zeros((ncrys, 4), dtype=dtype)
#    qarr[:,0] = 1. + R[:,0,0] + R[:,1,1] + R[:,2,2]
#
#    whr = (qarr[:,0] > 1.)
#    qarr[whr,1] = R[whr,2,1] - R[whr,1,2]
#    qarr[whr,2] = R[whr,0,2] - R[whr,2,0]
#    qarr[whr,3] = R[whr,1,0] - R[whr,0,1]
#
#    if whr.sum() < ncrys:
#        i, j, k = 0, 1, 2
#        qarr[~whr,i] = R[~whr,i, i] - (R[~whr,j, j] + R[~whr,k, k]) + 1.
#        qarr[~whr,j] = R[~whr,i, j] + R[~whr,j, i]
#        qarr[~whr,k] = R[~whr,k, i] + R[~whr,i, k]
#        qarr[~whr,3] = R[~whr,k, j] - R[~whr,j, k]
#
#        whr2 = ~whr * (R[:,1, 1] > R[:,0, 0])
#        i, j, k = 1, 2, 0
#        qarr[whr2,i] = R[whr2,i, i] - (R[whr2,j, j] + R[whr2,k, k]) + 1.
#        qarr[whr2,j] = R[whr2,i, j] + R[whr2,j, i]
#        qarr[whr2,k] = R[whr2,k, i] + R[whr2,i, k]
#        qarr[whr2,3] = R[whr2,k, j] - R[whr2,j, k]
#
#        whr3 = ~whr * (R[:,2, 2] > R[:,i, i]) # probably not correct if where whr2 isn't True
#        i, j, k = 2, 0, 1
#        qarr[whr2,i] = R[whr2,i, i] - (R[whr2,j, j] + R[whr2,k, k]) + 1.
#        qarr[whr2,j] = R[whr2,i, j] + R[whr2,j, i]
#        qarr[whr2,k] = R[whr2,k, i] + R[whr2,i, k]
#        qarr[whr2,3] = R[whr2,k, j] - R[whr2,j, k]
#
#        #qarr[~whr,:] = qarr[~whr,[3, 0, 1, 2]]
#
#        #qarr[~whr,1] = np.sqrt(0.5*(1. + R[~whr,0,0]) )
#        #qarr[~whr,2] = np.sqrt(0.5*(1. + R[~whr,1,1]) )
#        #qarr[~whr,3] = np.sqrt(0.5*(1. + R[~whr,2,2]) )
#
#    qarr *= 0.5 / np.sqrt(t)
#
#    return qarr

if __name__ == "__main__":
    import doctest
    doctest.testmod()

