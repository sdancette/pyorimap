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
    >>> qarr = np.array([[-1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,-1]]).astype(np.float32)
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

def q4_sym_cubic(static=True, dtype=np.float32):
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
    >>> np.allclose(ang, np.zeros(24, dtype=np.float32), atol=0.1)
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

def q4_sym_hex(static=True, dtype=np.float32):
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
    >>> np.allclose(ang, np.zeros(12, dtype=np.float32), atol=0.1)
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

def q4_sym_tetra(static=True, dtype=np.float32):
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
    >>> np.allclose(np.degrees(ang), np.zeros(8, dtype=np.float32), atol=0.1)
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

def q4_sym_ortho(static=True, dtype=np.float32):
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
    >>> np.allclose(np.degrees(ang), np.zeros(4, dtype=np.float32), atol=0.1)
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

def q4_random(n=1024, dtype=np.float32):
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
    >>> np.allclose(norm, np.ones(1024, dtype=np.float32))
    True
    """
    rand = np.random.rand(n,3)

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

def q4_from_axis_angle(axis, ang, dtype=np.float32):
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
    >>> np.allclose(q, np.array([1,0,0,0], dtype=np.float32))
    True
    >>> q = q4_from_axis_angle([[1,0,0],[1,1,1]], 0)
    >>> q.shape
    (2, 4)
    >>> q = q4_from_axis_angle([1,0,0], [0,1,2,3,4])
    >>> q.shape
    (5, 4)
    >>> norm = np.sqrt(np.sum(q**2, axis=1))
    >>> np.allclose(norm, np.ones(5, dtype=np.float32))
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

def q4_mult(qa, qb, dtype=np.float32):
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

def q4_inv(qarr, dtype=np.float32):
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
    >>> qa = np.array([0,1,0,0], dtype=np.float32)
    >>> qinv = q4_inv(qa)
    >>> np.allclose(qinv, np.array([0,-1,0,0], dtype=np.float32))
    True
    >>> qa = q4_random(n=1024)
    >>> qinv = q4_inv(qa)
    """

    qinv = qarr.copy()
    qinv[...,1] *=  -1
    qinv[...,2] *=  -1
    qinv[...,3] *=  -1

    return qinv

def q4_from_eul(eul, dtype=np.float32):
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

def q4_to_eul(qarr, dtype=np.float32):
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
    >>> eul = np.random.rand(1024,3).astype(np.float32)
    >>> eul[:,0] *= 360.; eul[:,1] *= 180.; eul[:,2] *= 360.
    >>> qarr = q4_from_eul(eul)
    >>> eulback = q4_to_eul(qarr)
    >>> np.allclose(eul, eulback, atol=0.1)
    True
    >>> eul = [[10,30,50],[10,0,0],[10,180,0]]
    >>> qarr = q4_from_eul(eul)
    >>> eulback = q4_to_eul(qarr)
    >>> np.allclose(eul, eulback, atol=0.1)
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

def mat_from_eul(eul, dtype=np.float32):
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

def mat_to_eul(R, dtype=np.float32):
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

def q4_from_mat(R, dtype=np.float32):
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
    >>> np.allclose(qarr, qback, atol=1e-5)
    True
    """
    qarr = q4_from_eul( mat_to_eul(R) )
    return qarr

def q4_to_mat(qarr, dtype=np.float32):
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

def q4_cosang2(qa, qb, dtype=np.float32):
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

def q4_disori_angle(qa, qb, qsym, method=1, return_index=False, dtype=np.float32):
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
    >>> np.allclose(ang, np.zeros(24, dtype=np.float32), atol=0.1)
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

def q4_disori_quat(qa, qb, qsym, frame='ref', method=1, return_index=False, dtype=np.float32):
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
    >>> qa = np.array([1,0,0,0], dtype=np.float32)
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
    >>> deso = np.arccos(q4_cosang2(qa, qavg))*2*180/np.pi
    >>> (deso < 0.1)
    True
    >>> np.allclose(qa, qavg, atol=1e-3)
    True
    """

    ii = 0
    theta= 999.
    nitermax = 10
    theta_iter = np.zeros(nitermax, dtype=DTYPEf) - 1.
    while (theta > 0.1) and (ii < nitermax):
        if ii == 0:
            # initialize avg orientation:
            qref = qarr[0,:]

        # disorientation of each crystal wrt average orientation:
        qdis = q4_disori_quat(qref, qarr, qsym, frame='crys_a', method=1)

        qtmp = np.sum(qdis, axis=0) # careful with np.float32 sum of very big arrays with more than 16*1024**2 quaternions
        qtmp /= np.sqrt(np.einsum('...i,...i', qtmp, qtmp))
        #qtmp /= np.sqrt(np.sum(qtmp**2)) # slower ?

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

    logging.info("Computed average grain orientation over {} crystals in {} iterations.".format(len(qarr), ii))
    logging.info("Theta convergence (degrees): {}".format(theta_iter))

    return qavg, GROD, GROD_stat, theta_iter

def q4_orispread(ncrys=1024, thetamax=1., misori=True, dtype=np.float32):
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
        The distribution of the misorientation angles is uniform only if misori=True

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
    rand = np.random.rand(ncrys, 3).astype(DTYPEf)
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

#def XXXq4_from_mat(R, dtype=np.float32):
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

