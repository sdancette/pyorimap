# -*- coding: utf-8 -*-

# pyorimap/orientation_map.py

"""
Perform orientation map analysis.

The module contains the following functions:

- `xxx(n)` - generate xxx
"""

import logging
import numpy as np
import cupy as cp
import pyvista as pv
from scipy import sparse

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

    Parameters
    ----------
    dimensions : array_like
        dimensions in the X, Y, Z, directions.
    spacing : float
        size of each (cubic) grid point.
    phase_to_crys : dict
        dictionary of crystal definition for each phase label (if multiphase).

    Attributes
    ----------
    xxx : float
        xxx.

    Methods
    -------
    get_neighborhood(connectivity=6)
        Construct the array of cell neighborhood.
    """

    def __init__(self, dimensions=(128+1,128+1,1+1), spacing=(1,1,1), phase_to_crys={1:Crystal(1,'no_name')}):
        """
        """
        super().__init__(dimensions=dimensions, spacing=spacing)
        self.phase_to_crys = phase_to_crys

    def get_neighborhood(self, connectivity=6):
        """
        Construct cell neighborhood information.
        In 2D (dim=2), connectivity = [4 | 8].
        In 3D (dim=3), connectivity = [6 | 26].
        """
        tol = np.array(self.spacing, dtype=DTYPEf).min()/100.
        whr = (self.points[:,0] < self.bounds[1]-tol)*(self.points[:,1] < self.bounds[3]-tol)*(self.points[:,2] < self.bounds[5]-tol)
        xyz = self.points[whr]
        x = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
        dx, dy, dz = self.spacing
        nn = self.n_cells

        # assessing 2D or 3D:
        if self.dimensions[2]-1 > 1:
            dim = 3
            if connectivity not in [6,26]:
                connectivity = 6
                logging.warning("Forcing connectivity to 6 (direct cell neighbors in 3D).")
        else:
            dim = 2
            if connectivity not in [4,8]:
                connectivity = 4
                logging.warning("Forcing connectivity to 4 (direct cell neighbors in 2D).")

        mydtype = [('icel', 'i4'), ('ineb', 'i4'), ('itype', 'i2'), ('order', 'u1'), ('val', 'f4')]
        self.neighborhood = np.rec.array( np.zeros(nn*int(connectivity / 2), dtype=mydtype) )
        icells = np.arange(0, nn, dtype=np.int32)

        # East-West:
        n1 = 0; n2 = nn
        self.neighborhood.icel[n1:n2] = icells
        self.neighborhood.ineb[n1:n2] = xyz_to_index(x+dx, y, z, mode='cells', grid=self)
        self.neighborhood.itype[n1:n2] = 1
        self.neighborhood.order[n1:n2] = 1

        # North-South:
        n1 = nn; n2 = 2*nn
        self.neighborhood.icel[n1:n2] = icells
        self.neighborhood.ineb[n1:n2] = xyz_to_index(x, y+dy, z, mode='cells', grid=self)
        self.neighborhood.itype[n1:n2] = 2
        self.neighborhood.order[n1:n2] = 1

        if dim == 3:
            # Bottom-Top:
            n1 = 2*nn; n2 = 3*nn
            self.neighborhood.icel[n1:n2] = icells
            self.neighborhood.ineb[n1:n2] = xyz_to_index(x, y, z+dz, mode='cells', grid=self)
            self.neighborhood.itype[n1:n2] = 3
            self.neighborhood.order[n1:n2] = 1

        if connectivity >= 8:
            # 2D, in plane diagonal neighbors:
            # South-East
            if dim == 2:
                n1 = 2*nn; n2 = 3*nn
            else:
                n1 = 3*nn; n2 = 4*nn
            self.neighborhood.icel[n1:n2] = icells
            self.neighborhood.ineb[n1:n2] = xyz_to_index(x+dx, y+dy, z, mode='cells', grid=self)
            self.neighborhood.itype[n1:n2] = 4
            self.neighborhood.order[n1:n2] = 2

            # South-West
            if dim == 2:
                n1 = 3*nn; n2 = 4*nn
            else:
                n1 = 4*nn; n2 = 5*nn
            self.neighborhood.icel[n1:n2] = icells
            self.neighborhood.ineb[n1:n2] = xyz_to_index(x-dx, y+dy, z, mode='cells', grid=self)
            self.neighborhood.itype[n1:n2] = 5
            self.neighborhood.order[n1:n2] = 2

        if connectivity == 26:
            # Bottom-East
            pass

            # Bottom-West
            pass

            # Bottom-North
            pass

            # Bottom-South
            pass

            # 4 cube diagonals
            pass

        whr = (self.neighborhood.ineb >= 0)*(self.neighborhood.ineb < nn)
        self.neighborhood = self.neighborhood[whr]

    def compute_neighbor_disori(self, mode='numba_cpu'):
        """
        Compute the disorientation between neighboring cells.
        """
        try:
            ncrys = len(self.qarray)
        except AttributeError:
            self.qarray = q4np.q4_from_eul(self.cell_data['eul'])

        try:
            nneb = len(self.neighborhood)
        except AttributeError:
            logging.warning("Computing neighborhood with direct cell connectivity. Compute neighborhood behorehand for greater control.")
            if self.dimensions[2]-1 > 1: # 3D
                self.get_neighborhood(connectivity=6)
            else: # 2D
                self.get_neighborhood(connectivity=4)

        logging.info("Starting to compute neighbor disorientation.")
        for phi in self.phase_to_crys.keys():
            # restrict neighborhood to identical phase on the 2 sides:
            phase = self.cell_data['phase']
            icel = self.neighborhood.icel
            ineb = self.neighborhood.ineb
            whr = (phase[icel] == phi) * (phase[icel] == phase[ineb])

            if phi > 0:
                qa = self.qarray[icel[whr]]
                qb = self.qarray[ineb[whr]]
                try:
                    qsym = self.phase_to_crys[phi].qsym
                except AttributeError:
                    self.phase_to_crys[phi].infer_symmetry()
                    self.phase_to_crys[phi].get_qsym()
                    qsym = self.phase_to_crys[phi].qsym

                self.neighborhood.val[whr] = q4nc.q4_disori_angle(qa, qb, qsym, method=1)

            elif phi == 0:
                self.neighborhood.val[whr] = -99
        logging.info("Finished to compute neighbor disorientation.")

    def sparse_connected_components(self, thres=10.):
        """
        Compute the connected components based on cell neighborhood.
        """
        ll = self.n_cells
        whr = (self.neighborhood.val < thres)
        ipix2 = np.append(self.neighborhood.icel[whr], self.neighborhood.ineb[whr], axis=0)
        ineb2 = np.append(self.neighborhood.ineb[whr], self.neighborhood.icel[whr], axis=0)
        val2 = np.tile(self.neighborhood.val[whr], 2)

        A = sparse.csr_array((val2, (ipix2, ineb2)),
                             shape=(ll,ll), dtype=np.float32)

        ncomp, labels = sparse.csgraph.connected_components(A, directed=False, return_labels=True)

        self.cell_data['grains'] = labels + 1 # starting at 1 instead of 0


def read_from_ctf(filename, dtype=[('phase', 'u1'), ('X', 'f4'), ('Y', 'f4'),
                                   ('Bands', 'i2'), ('Error', 'i2'),
                                   ('phi1', 'f4'), ('Phi', 'f4'), ('phi2', 'f4'),
                                   ('MAD', 'f4'), ('BC', 'i2'), ('BS', 'i2')]):
    """
    Read a ascii .ctf file and return an OriMap object
    with data stored at the cell centers.
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
        crys.get_qsym()
        phase_to_crys[phase] = crys

    logging.info("Starting to read data from{}".format(filename))

    ctfdata = np.genfromtxt(filename, skip_header=nskip, dtype=dtype)
    if ctfdata['phase'].min() == 0:
        phase_to_crys[0] = Crystal(0, name='unindexed', abc=np.zeros(3)*np.NaN, ang=np.zeros(3)*np.NaN)

    # pyvista Image object:
    logging.info("Generating pyvista ImageData object.")

    orimap = OriMap((XCells+1, YCells+1, 2), (XStep, YStep, max(XStep, YStep)),
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

def xyz_to_index(x, y, z, grid, mode='cells'):
    """
    Return the cell/point indices in the grid from (x,y,z) coordinates.
    """
    if mode == 'cells':
        nx, ny, nz = grid.dimensions[0]-1, grid.dimensions[1]-1, grid.dimensions[2]-1
    elif mode == 'points':
        nx, ny, nz = grid.dimensions[0], grid.dimensions[1], grid.dimensions[2]
    dx, dy, dz = grid.spacing
    ox, oy, oz = grid.bounds[0], grid.bounds[2], grid.bounds[4]

    checkxyz = (x >= 0)*(x < nx)*(y >= 0)*(y < ny)*(z >= 0)*(z < nz)

    a = np.int32(np.round((z-oz)/dz)*nx*ny)
    b = np.int32(np.round((y-oy)/dy)*nx)
    c = np.int32(np.round((x-ox)/dx))
    ii = a + b + c
    ii[~checkxyz] = -99999
    return ii

def index_to_xyz(ii, grid, mode='cells'):
    """
    Return (x,y,z) coordinates in the grid from cell/point indices.
    """
    if mode == 'cells':
        nx, ny, nz = grid.dimensions[0]-1, grid.dimensions[1]-1, grid.dimensions[2]-1
    elif mode == 'points':
        nx, ny, nz = grid.dimensions[0], grid.dimensions[1], grid.dimensions[2]
    dx, dy, dz = grid.spacing
    ox, oy, oz = grid.bounds[0], grid.bounds[2], grid.bounds[4]

    quo1, rem1 = np.divmod(ii, nx*ny)
    z = quo1*dz + oz
    quo2, rem2 = np.divmod(rem1, nx)
    y = quo2*dy + oy
    x = rem2*dx + ox
    return np.column_stack((x,y,z))
