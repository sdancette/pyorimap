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

from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

logging.basicConfig(filename='orimap.log', level=logging.INFO, format='%(asctime)s %(message)s')

DTYPEf = np.float32
DTYPEi = np.int32

class Crystal:
    def __init__(self, phaseid, abc=[1,1,1], ang=[90,90,90], name='no_name'):
        """
        Initialize a Crystal object.
        """
        self.phase = int(phaseid)
        self.abc = np.atleast_1d(abc).astype(np.float32)
        self.ang = np.atleast_1d(ang).astype(np.float32)
        self.name = name

        # infer crystal symmetry:
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

    def __str__(self):
        return f"{self.phase}, {self.name}, {self.abc}, {self.ang}: {self.sym}"

    def __repr__(self):
        return (f"{type(self).__name__}"
                f'(phase={self.phase}, '
                f'name="{self.name}", '
                f"abc={self.abc}, "
                f"ang={self.ang}, "
                f'sym="{self.sym}")')


def read_from_ctf(filename, dtype=[('phase', 'u1'), ('X', 'f4'), ('Y', 'f4'),
                                   ('Bands', 'i2'), ('Error', 'i2'),
                                   ('phi1', 'f4'), ('Phi', 'f4'), ('phi2', 'f4'),
                                   ('MAD', 'f4'), ('BC', 'i2'), ('BS', 'i2')]):
    """
    Read a ascii .ctf file and return a pyvista
    ImageData object with data stored at the cell centers.
    """
    logging.info("Starting to read {}".format(filename))
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
        phase_to_crys[phase] = Crystal(phase, abc, ang, name)

    ctfdata = np.genfromtxt(filename, skip_header=nskip, dtype=dtype)
    if ctfdata['phase'].min() == 0:
        phase_to_crys[0] = Crystal(0, abc=[np.NaN,np.NaN,np.NaN], ang=[np.NaN,np.NaN,np.NaN], name='unindexed')

    # pyvista Image object:
    logging.info("Generating pyvista ImageData object.")

    grid = pv.ImageData(dimensions=(XCells+1, YCells+1, 2),
                        spacing=(XStep, YStep, 1),
                        origin=(0, 0, 0) )
    grid.cell_data['phase'] = ctfdata['phase']
    grid.cell_data['Bands'] = ctfdata['Bands']
    grid.cell_data['Error'] = ctfdata['Error']
    grid.cell_data['MAD'] = ctfdata['MAD']
    grid.cell_data['BC'] = ctfdata['BC']
    grid.cell_data['BS'] = ctfdata['BS']
    grid.cell_data['eul'] = structured_to_unstructured(ctfdata[['phi1', 'Phi', 'phi2']])
    grid.qarray = q4np.q4_from_eul(grid.cell_data['eul'])
    grid.phase_to_crys = phase_to_crys

    fvtk = filename[:-4]+'.vtk'
    logging.info("Saving vtk file: {}".format(fvtk))
    grid.save(fvtk)

    return grid

