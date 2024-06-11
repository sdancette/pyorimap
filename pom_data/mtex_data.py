# -*- coding: utf-8 -*-

import pkgutil
import logging
import numpy as np

from io import StringIO

DTYPEf = np.float32
DTYPEi = np.int32

def load_mtex_qsym(sym='cubic', dtype=np.float32):
    """
    Load equivalent quaternion array generated by mtex for the given crystal symmetry.
    (Intended use for doctest.)
    """
    if sym == 'cubic':
        data = pkgutil.get_data('pom_data', 'qsym_cubic_mtex.txt')
    elif sym == 'hex':
        data = pkgutil.get_data('pom_data', 'qsym_hex_mtex.txt')
    elif sym == 'tetra':
        data = pkgutil.get_data('pom_data', 'qsym_tetra_mtex.txt')
    elif sym == 'ortho':
        data = pkgutil.get_data('pom_data', 'qsym_ortho_mtex.txt')
    data = StringIO(data.decode('utf-8'))
    qsym = np.loadtxt(data, delimiter=',', dtype=dtype)

    return qsym