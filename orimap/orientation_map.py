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
import skimage as sk
from scipy import sparse

import quaternions_np as q4np
import quaternions_cp as q4cp
import quaternions_numba_cpu as q4nc
import virtual_micro as vmic

from dataclasses import dataclass, field
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

#logging.basicConfig(filename='orimap.log', level=logging.INFO, format='%(asctime)s %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DTYPEf = np.float32
DTYPEi = np.int32

@dataclass
class Crystal:
    phase: int = 1
    name: str = 'no_name'
    abc: np.ndarray[(int,), np.dtype[DTYPEf]] = field(default_factory=lambda: np.ones(shape=(3,), dtype=DTYPEf))
    ang: np.ndarray[(int,), np.dtype[DTYPEf]] = field(default_factory=lambda: np.ones(shape=(3,), dtype=DTYPEf)*90)
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

class StructuringElement:
    """
    Structuring Element of given radius and connectivity.

    Parameters
    ----------
    radius : int, default: 1
        Radius of the structuring element.
    connectivity : str, default: 'face'
        Type of connectivity ('face', 'edge' or 'vertice').
    dim : int
        Dimension of the underlying grid.
    """

    def __init__(self, radius=1, connectivity='face', dim=3):
        logging.info("Generating structuring element with radius {} and {}-type connectivity in dimension {}.".format(radius, connectivity, dim))

        self.r = radius
        self.connec = connectivity
        self.dim = dim

        if self.connec == 'face':
            self.selem = sk.morphology.ball(self.r).astype(np.uint8)
        else: # full connectivity including edge and vertice neighbors
            w = 2*self.r + 1
            self.selem = np.ones((w,w,w), dtype=np.uint8)

        dxyz = np.column_stack(np.nonzero(self.selem))[:,::-1] - self.r # equivalent to np.column_stack(np.where(selem>0)) - r
        # restrict to neighbors in 1 of the 2 opposite directions
        # (relation in the other direction will be evaluated from the other cell):
        nneb = int((len(dxyz)-1)/2)
        dxyz = dxyz[nneb+1:]

        if self.dim == 2:
            # keep only in-plane neighbors (at the same 'z' position):
            dxyz = dxyz[dxyz[:,2]==0]
            self.selem = self.selem[self.r]

        self.dxyz = dxyz
        self.nneb = len(dxyz)
        logging.info("selem: {}".format(self.selem))
        logging.info("Relative position of neighbors: {}.".format(self.dxyz))
        logging.info("Half-number of neighbors to be considered: {}.".format(self.nneb))


class OriMap(pv.ImageData):
    """
    OriMap class (inheriting from pyvista ImageData).

    Data is stored at the cell centers (as cell_data) by default rather than at the vertices.

    Parameters
    ----------
    uinput : str, vtk.vtkImageData, pyvista.ImageData, optional
        Filename or dataset to initialize the uniform grid from.
        If set, dimensions and spacing are ignored.
    dimensions : sequence[int], optional
        Dimensions of the uniform grid in the (X, Y, Z) directions.
    spacing : sequence[float], default: (1.0, 1.0, 1.0)
        Spacing of the uniform grid in each dimension. Must be positive.
    phase_to_crys : dict
        Dictionary pointing to the crystal object from the phase ID.

    Attributes
    ----------
    filename : str
        String corresponding to the filename.

    Methods
    -------
    get_neighborhood(connectivity=6)
        Construct the array of cell neighborhood.
    """

    def __init__(self, uinput=None, dimensions=None, spacing=(1.,1.,1.), phase_to_crys={}):
        """
        """
        super().__init__(uinput, dimensions=dimensions, spacing=spacing)
        self.phase_to_crys = phase_to_crys
        self.filename = 'no_name.vtk'

        # assessing 2D or 3D:
        if self.dimensions[2]-1 > 1:
            self.dim3D = True
        else:
            self.dim3D = False

    def xyz_to_index(self, x, y, z, mode='cells', method=1):
        """
        Return the cell/point indices in the grid from (x,y,z) coordinates.
        """
        if mode == 'cells':
            nx, ny, nz = self.dimensions[0]-1, self.dimensions[1]-1, self.dimensions[2]-1
        elif mode == 'points':
            nx, ny, nz = self.dimensions[0], self.dimensions[1], self.dimensions[2]
        dx, dy, dz = self.spacing
        ox, oy, oz = self.bounds[0], self.bounds[2], self.bounds[4]

        x = np.round((x-ox)/dx).astype(DTYPEi)
        y = np.round((y-oy)/dy).astype(DTYPEi)
        z = np.round((z-oz)/dz).astype(DTYPEi)
        checkxyz = (x >= 0)*(x < nx)*(y >= 0)*(y < ny)*(z >= 0)*(z < nz)

        if method == 1:
            z *= nx*ny
            y *= nx
            ii = z + y + x
            ii[~checkxyz] = -1
        else:
            ii = np.zeros(len(x), dtype=DTYPEi) - 1
            x = x[checkxyz]
            y = y[checkxyz]
            z = z[checkxyz]
            ii[checkxyz] = np.ravel_multi_index([z,y,x],(nz,ny,nx))
        return ii

    def index_to_xyz(self, ii, mode='cells', method=1):
        """
        Return (x,y,z) coordinates in the grid from cell/point indices.
        """
        if mode == 'cells':
            nx, ny, nz = self.dimensions[0]-1, self.dimensions[1]-1, self.dimensions[2]-1
        elif mode == 'points':
            nx, ny, nz = self.dimensions[0], self.dimensions[1], self.dimensions[2]
        dx, dy, dz = self.spacing
        ox, oy, oz = self.bounds[0], self.bounds[2], self.bounds[4]

        if method == 1:
            quo1, rem1 = np.divmod(ii, nx*ny)
            z = quo1*dz + oz
            quo2, rem2 = np.divmod(rem1, nx)
            y = quo2*dy + oy
            x = rem2*dx + ox
        else:
            z,y,x = np.unravel_index(ii, (nz,ny,nx))
            z = z*dz + oz
            y = y*dy + oy
            x = x*dx + ox

        return np.column_stack((x,y,z)).astype(DTYPEf)

    def get_neighborhood(self, radius=1, connectivity='face'):
        """
        Get neighbors with given connectivity and corresponding disorientation.
        """
        logging.info("Starting to retrieve neighbors with {} connectivity and radius {}.".format(connectivity, radius))

        # get the coordinates of the cells (dimensions - 1 wrt the points):
        tol = np.array(self.spacing, dtype=DTYPEf).min()/100.
        self.whrC = (self.points[:,0] < self.bounds[1]-tol)*\
                    (self.points[:,1] < self.bounds[3]-tol)*\
                    (self.points[:,2] < self.bounds[5]-tol)
        xyzC = self.points[self.whrC]
        #x = xyzC[:,0]; y = xyzC[:,1]; z = xyzC[:,2]
        dx, dy, dz = self.spacing
        nn = self.n_cells

        # structuring element to define the connectivity of cells:
        dim = 3 if self.dim3D else 2
        self.selem = StructuringElement(radius, connectivity, dim)

        # loop over the different types of neighbors defined by the structuring element:
        cellid = np.arange(nn).astype(DTYPEi)
        self.deso = np.zeros((nn,self.selem.nneb), dtype=DTYPEf) + 3000.
        self.neighbors = np.zeros((nn,self.selem.nneb), dtype=DTYPEi)

        logging.info("Starting to compute neighbor disorientation.")

        for ineb, dxyz in enumerate(self.selem.dxyz):
            self.neighbors[:,ineb] = self.get_neighbors_for_ineb(xyzC, dxyz, ineb)

            self.get_disori_for_ineb(cellid, ineb, mode='numba_cpu')

        logging.info("Finished to compute neighbor disorientation.")

    def build_cell_graph(self, thres=10., symmetric=False):
        """
        Build the graph of cell connections as a scipy.sparse csr_array.
        """
        logging.info("Starting to build cell graph.")

        nn = self.n_cells
        cellid = np.arange(nn).astype(DTYPEi)

        whr = np.where((self.neighbors >= 0)*(self.deso < thres))
        self.Adjmat = sparse.csr_array((self.deso[whr], (cellid[whr[0]], self.neighbors[whr])),
                                       shape=(nn,nn), dtype=DTYPEf)
        if symmetric:
            self.Adjmat += self.Adjmat.T

        logging.info("Finished to build cell graph.")

    def get_connected_components(self):
        """
        Get the connected components based on given threshold.
        """
        ncomp, labels = sparse.csgraph.connected_components(self.Adjmat, directed=False, return_labels=True)

        self.cell_data['region'] = labels + 1 # starting at 1 instead of 0

        logging.info("Finished to compute {} connected components.".format(ncomp))

    def get_neighbors_for_ineb(self, xyzC, dxyz, ineb):
        """
        Construct cell neighborhood information for the ineb_th family of neighbors.
        """

        dz = dxyz[2]; dy = dxyz[1];  dx = dxyz[0]
        z = xyzC[:,2]; y = xyzC[:,1]; x = xyzC[:,0]

        ii = self.xyz_to_index(x+dx, y+dy, z+dz, mode='cells')
        return ii

    def get_disori_for_ineb(self, icel, ineb, mode='numba_cpu'):
        """
        Compute the disorientation with the ineb_th family of neighbors.
        NB1: disorientation is initialized at +3000. for cell neighbors outside of the grid bounds.
        NB2: disorientation is initialized at +2000. for cell neighbors belonging to different phases.
        NB3: disorientation is initialized at -1. for cell neighbors belonging to phase 0 (unindexed by default).
        """
        phase = self.cell_data['phase']

        neighb = self.neighbors[:,ineb]
        # restrict to existing neighbors, i.e. within the bounds of the grid:
        whrNeb = (neighb >= 0)

        try:
            ncrys = len(self.qarray)
        except AttributeError:
            self.qarray = q4np.q4_from_eul(self.cell_data['eul'])

        self.deso[:,ineb][whrNeb] = 2000.
        for phi in self.phase_to_crys.keys():
            # restrict neighborhood to existing neighbors and identical phase on the 2 sides:
            whr = whrNeb * (phase[icel] == phi) * (phase[icel] == phase[neighb])

            if phi > 0:
                qa = self.qarray[icel[whr]]
                qb = self.qarray[neighb[whr]]
                try:
                    qsym = self.phase_to_crys[phi].qsym
                except AttributeError:
                    self.phase_to_crys[phi].infer_symmetry()
                    self.phase_to_crys[phi].get_qsym()
                    qsym = self.phase_to_crys[phi].qsym

                self.deso[:,ineb][whr] = q4nc.q4_disori_angle(qa, qb, qsym, method=1)

            elif phi == 0:
                self.deso[:,ineb][whr] = -1.

    def filter_grains(self, nmin=1, nmax=2**31, phimin=1, phimax=2**8):
        """
        Relabel grains in a consecutive sequence after the exclusion of regions outside of ncell_range and phase_range.
        """
        phase = self.cell_data['phase']
        region = self.cell_data['region']

        # first, exclusion of regions outside of phase_min and phase_max:
        whr = (phase < phimin) + (phase >= phimax)
        region[whr] = 0
        unic, indices, counts = np.unique(region, return_inverse=True, return_counts=True)

        # then exclusion of regions outside of npix_min and npix_max:
        toremove = (counts < nmin) + (counts >= nmax)
        unic[toremove] = 0
        region = unic[indices]
        # re-run the counting:
        unic, indices, counts = np.unique(region, return_inverse=True, return_counts=True)

        # redefine the unique grain label in a consecutive sequence:
        newu = np.arange(len(unic))
        if unic.min() == 0:
            self.cell_data['grains'] = newu[indices] # preserve '0' region with label 0
        elif unic.min() >= 1:
            self.cell_data['grains'] = newu[indices]+1 # there is no region outside of the specified ranges, start labeling at 1
        # final count:
        unic, counts = np.unique(self.cell_data['grains'], return_counts=True)

        return unic, counts

    def save_phase_info(self):
        """
        Save the properties of individual phases and crystals to .phi file.
        """
        try:
            phases = list(self.phase_to_crys.keys())
        except AttributeError:
            logging.error("No phase data to save.")

        mydtype=[('name', 'U15'), ('phase', 'i2'), ('sym', 'U10'),
                ('a', 'f4'), ('b', 'f4'), ('c', 'f4'),
                ('alpha', 'f4'), ('beta', 'f4'), ('gamma', 'f4')]
        fmt = ['%s', '%4d', '%s',
               '%8.4f', '%8.4f', '%8.4f',
               '%6.1f', '%6.1f', '%6.1f']
        crys = []
        for phi in sorted(phases):
            thephase = self.phase_to_crys[phi]
            crys.append( (thephase.name, thephase.phase, thephase.sym,
                          thephase.abc[0], thephase.abc[1], thephase.abc[2],
                          thephase.ang[0], thephase.ang[1], thephase.ang[2]) )
        crys = np.array(crys, dtype=mydtype)
        filename = self.filename[:-4]+'.phi'
        np.savetxt(filename, crys, delimiter=',', fmt=fmt, header=str(crys.dtype.names)[1:-1])

    def read_phase_info(self, f=None):
        """
        Read the properties of individual phases and crystals from .phi file.
        """
        if f is None:
            f = self.filename[:-4]+'.phi'

        nophasedata = True
        try:
            thephases = np.atleast_1d( np.unique(self.cell_data['phase']) )
            nophasedata = False
        except KeyError:
            logging.warning("Phase data wasn't found within available cell_data keys: {}".format(self.cell_data.keys()))
            nophasedata = True

        if nophasedata:
            logging.warning("Assuming homogeneous 'phase1' with cubic symmetry for the whole microstructure.")
            self.phase_to_crys[1] = Crystal(1, name='phase1', sym='cubic')
            self.cell_data['phase'] = 1
        else:
            mydtype=[('name', 'U15'), ('phase', 'u1'), ('sym', 'U10'),
                    ('a', 'f4'), ('b', 'f4'), ('c', 'f4'),
                    ('alpha', 'f4'), ('beta', 'f4'), ('gamma', 'f4')]

            readPhase = True
            try:
                phases = np.atleast_1d(np.rec.array(np.genfromtxt(f, delimiter=',', dtype=mydtype)))
                if not np.allclose(np.sort(phases.phase), thephases):
                    readPhase = False
                    logging.error("Phase IDs do not match in cell_data['phase'] and {}: {}, {}".format(f, thephases, phases.phase))
            except FileNotFoundError:
                readPhase = False
                logging.error("Failed to read .phi phase file {}.".format(filename))

            #self.phase_to_crys = dict()
            if readPhase:
                for phi in phases:
                    self.phase_to_crys[phi.phase] = Crystal(phi.phase, name=phi.name, sym=phi.sym,
                                                            abc=np.array([phi.a,phi.b,phi.c,]),
                                                            ang=np.array([phi.alpha,phi.beta,phi.gamma,]))
            else:
                logging.warning("Assuming cubic symmetry for all phases ({}).".format(thephases))
                for phi in thephases:
                    self.phase_to_crys[phi] = Crystal(phi, name='phase'+str(phi), sym='cubic')

def read_from_ctf(filename, dtype=None):
    """
    Read a ascii .ctf file and return an OriMap object
    with data stored at the cell centers.
    """
    if dtype is None:
        dtype = [('phase', 'u1'), ('X', 'f4'), ('Y', 'f4'),
                ('Bands', 'i2'), ('Error', 'i2'),
                ('phi1', 'f4'), ('Phi', 'f4'), ('phi2', 'f4'),
                ('MAD', 'f4'), ('BC', 'i2'), ('BS', 'i2')]

    logging.info("Reading header in {}.".format(filename))
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

    logging.info("Reading data from {}.".format(filename))
    logging.info("Column names and types: {}".format(dtype))

    ctfdata = np.genfromtxt(filename, skip_header=nskip, dtype=dtype)
    if ctfdata['phase'].min() == 0:
        phase_to_crys[0] = Crystal(0, name='unindexed', abc=np.zeros(3)*np.NaN, ang=np.zeros(3)*np.NaN)

    # pyvista Image object:
    logging.info("Generating pyvista ImageData object.")

    orimap = OriMap(None, (XCells+1, YCells+1, 2), (XStep, YStep, max(XStep, YStep)),
                    phase_to_crys )

    orimap.cell_data['phase'] = ctfdata['phase']
    orimap.cell_data['Bands'] = ctfdata['Bands']
    orimap.cell_data['Error'] = ctfdata['Error']
    orimap.cell_data['MAD'] = ctfdata['MAD']
    orimap.cell_data['BC'] = ctfdata['BC']
    orimap.cell_data['BS'] = ctfdata['BS']
    orimap.cell_data['eul'] = structured_to_unstructured(ctfdata[['phi1', 'Phi', 'phi2']])
    orimap.qarray = q4np.q4_from_eul(orimap.cell_data['eul'])

    orimap.filename = filename
    fvtk = filename[:-4]+'.vtk'
    logging.info("Saving vtk file: {}".format(fvtk))
    orimap.save(fvtk)
    orimap.save_phase_info()

    return orimap

def read_from_vtk(filename):
    """
    Read a .vtk file and return an OriMap object.
    """
    logging.info("Reading data from {}.".format(filename))

    orimap = OriMap(uinput=filename, phase_to_crys={})
    orimap.filename = filename
    orimap.read_phase_info()

    phases = np.unique(orimap.cell_data['phase'])
    # crystal definition:
    propercrys = True
    keys = np.sort( np.array( list(orimap.phase_to_crys.keys()) ) )
    if not np.allclose(phases, keys):
        propercrys = False
        logging.warning("Crystal dictionary not specified properly. Default crystals will be attributed to the phases.")

    if not propercrys:
        for iphase, phase in enumerate(phases):
            orimap.phase_to_crys[phase] = Crystal(phase, name='phase'+str(phase))
            logging.info("{}".format(orimap.phase_to_crys[phase]))
    else:
        for iphase, phase in enumerate(phases):
            logging.info("{}".format(orimap.phase_to_crys[phase]))

    logging.info("Finished reading from {}.".format(filename))

    return orimap


