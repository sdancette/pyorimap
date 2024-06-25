# -*- coding: utf-8 -*-

# pyorimap/orimap.py

"""
Perform orientation map analysis.

The module contains the following functions:

- `xxx(n)` - generate xxx
"""

import logging
import numpy as np
import cupy as cp
import pyvista as pv

import quaternions_np as q4np
import quaternions_cp as q4cp
import quaternions_numba_cpu as q4nc
import virtual_micro as vmic

from dataclasses import dataclass, field
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

#logging.basicConfig(filename='orimap.log', level=logging.INFO, format='%(asctime)s %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

DTYPEf = np.float32
DTYPEi = np.int32

@dataclass
class Crystal:
    phase: int = 1
    name: str = 'no_name'
    abc: np.ndarray[(int,), np.dtype[np.float32]] = field(default_factory=lambda: np.ones(shape=(3,), dtype=np.float32))
    ang: np.ndarray[(int,), np.dtype[np.float32]] = field(default_factory=lambda: np.ones(shape=(3,), dtype=np.float32)*90)
    sym: str = 'none'

    def infer_symmetry(self):
        """
        Infer crystal symmetry from the `abc` and `ang` vectors.
        """
        if (self.ang[0]==90.) and (self.ang[1]==90.) and (self.ang[2]==90.):
            # cubic, tetra or orthorhombic
            if (self.abc[0]==self.abc[1]) and (self.abc[0]==self.abc[2]): # cubic
                self.sym = 'cubic'
            elif (self.abc[0]==self.abc[1]): # tetragonal
                self.sym = 'tetra'
            else: # orthorhombic
                self.sym = 'ortho'
        elif (self.ang[0]==90.) and (self.ang[1]==90.) and (self.ang[2]==120.):
            if (self.abc[0]==self.abc[1]): # hexagonal
                self.sym = 'hex'
            else:
                logging.warning("Unexpected crystal lattice")
                self.sym = 'none'
        else:
            self.sym = 'none'
            if not name == 'unindexed':
                logging.warning("Unexpected crystal lattice")

    def get_qsym(self):
        """
        Compute the (n, 4) quaternion array for the present crystal symmetry.
        """
        if self.sym == 'cubic':
            self.qsym = q4np.q4_sym_cubic()
        elif self.sym == 'hex':
            self.qsym = q4np.q4_sym_hex()
        elif self.sym == 'tetra':
            self.qsym = q4np.q4_sym_tetra()
        elif self.sym == 'ortho':
            self.qsym = q4np.q4_sym_ortho()
        else:
            logging.warning("No quaternion symmetry operator implemented for crystal {self.sym}.")

class OriMap(pv.ImageData):
    """
    OriMap class inheriting from pyvista ImageData.
    """
    def __init__(self, dimensions=(128+1,128+1,1+1), spacing=(1,1,1), phase_to_crys={1:Crystal(1,'no_name')}):
        super().__init__(dimensions=dimensions, spacing=spacing)
        self.phase_to_crys = phase_to_crys


def read_from_ctf(filename, dtype=[('phase', 'u1'), ('X', 'f4'), ('Y', 'f4'),
                                   ('Bands', 'i2'), ('Error', 'i2'),
                                   ('phi1', 'f4'), ('Phi', 'f4'), ('phi2', 'f4'),
                                   ('MAD', 'f4'), ('BC', 'i2'), ('BS', 'i2')]):
    """
    Read a ascii .ctf file and return an OriMap object
    (inheriting from pyvista ImageData object) with data stored at the cell centers.
    """
    logging.info("Starting to read header in {}".format(filename))
    nlinemax = 30
    with open(filename, 'r') as f:
        head = [next(f) for _ in range(nlinemax)]
    for iline in range(nlinemax):
        line = head[iline]
        if 'JobMode' in line:
            data = line.split()
            JobMode = data[1]
        elif 'XCells' in line:
            data = line.split()
            XCells = int(data[1])
        elif 'YCells' in line:
            data = line.split()
            YCells = int(data[1])
        elif 'XStep' in line:
            data = line.split()
            XStep = float(data[1])
        elif 'YStep' in line:
            data = line.split()
            YStep = float(data[1])
        elif 'AcqE1' in line:
            data = line.split()
            AcqE1 = float(data[1])
        elif 'AcqE2' in line:
            data = line.split()
            AcqE2 = float(data[1])
        elif 'AcqE3' in line:
            data = line.split()
            AcqE3 = float(data[1])
        elif 'Euler angles refer to' in line:
            data = line.split()
            EulerFrame = data[4:8]
        elif 'Phases' in line:
            data = line.split()
            nPhases = int(data[1])
            nskip = iline + 1 + nPhases + 1
            phaseStart = iline + 1
            phaseEnd = iline + 1 + nPhases
        elif 'Bands' in line:
            break

    phase_to_crys = dict()
    for iphase in range(nPhases):
        phase = iphase + 1
        data = head[phaseStart+iphase].split('\t')
        abc = [float(v) for v in data[0].split(';')]
        ang = [float(v) for v in data[1].split(';')]
        name = data[2]
        crys = Crystal(phase, name, abc, ang)
        crys.infer_symmetry()
        phase_to_crys[phase] = crys

    logging.info("Starting to read data from{}".format(filename))

    ctfdata = np.genfromtxt(filename, skip_header=nskip, dtype=dtype)
    if ctfdata['phase'].min() == 0:
        phase_to_crys[0] = Crystal(0, name='unindexed', abc=np.zeros(3)*np.NaN, ang=np.zeros(3)*np.NaN)

    # pyvista Image object:
    logging.info("Generating pyvista ImageData object.")

    orimap = OriMap((XCells+1, YCells+1, 2), (XStep, YStep, 1),
                    phase_to_crys )
    orimap.cell_data['phase'] = ctfdata['phase']
    orimap.cell_data['Bands'] = ctfdata['Bands']
    orimap.cell_data['Error'] = ctfdata['Error']
    orimap.cell_data['MAD'] = ctfdata['MAD']
    orimap.cell_data['BC'] = ctfdata['BC']
    orimap.cell_data['BS'] = ctfdata['BS']
    orimap.cell_data['eul'] = structured_to_unstructured(ctfdata[['phi1', 'Phi', 'phi2']])
    orimap.qarray = q4np.q4_from_eul(orimap.cell_data['eul'])

    fvtk = filename[:-4]+'.vtk'
    logging.info("Saving vtk file: {}".format(fvtk))
    orimap.save(fvtk)

    return orimap

