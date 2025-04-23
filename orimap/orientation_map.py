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
import scipy.ndimage as ndi
from scipy import sparse

from pyorimap.quaternions import quaternions_np as q4np
from pyorimap.quaternions import quaternions_cp as q4cp
from pyorimap.quaternions import quaternions_numba_cpu as q4nCPU
from pyorimap.quaternions import quaternions_numba_gpu as q4nGPU
from pyorimap.orimap import virtual_micro as vmic

from dataclasses import dataclass, field
from typing import List, Tuple
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from numba import njit, prange, int8, int32, float32
from numba.types import Tuple as nTuple

#logging.basicConfig(filename='orimap.log', level=logging.INFO, format='%(asctime)s %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

DTYPEf = np.float32
DTYPEi = np.int32

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

    def __init__(self):
        self.radius = None
        self.connectivity = None
        self.dim = None

    def set_selem(self, radius=1, connectivity='face', dim=3):
        #logging.info("Generating structuring element with radius {} and {}-type connectivity in dimension {}.".format(radius, connectivity, dim))

        self.radius = radius
        self.connectivity = connectivity
        self.dim = dim

        if self.connectivity == 'face':
            self.selem = sk.morphology.ball(self.radius).astype(np.uint8)
        else: # full connectivity including edge and vertice neighbors
            w = 2*self.radius + 1
            self.selem = np.ones((w,w,w), dtype=np.uint8)

        dxyz = np.column_stack(np.nonzero(self.selem))[:,::-1] - self.radius # equivalent to np.column_stack(np.where(selem>0)) - r
        # restrict to neighbors in one of the two opposite directions
        # (relation in the other direction will be evaluated from the other cell):
        nneb = int((len(dxyz)-1)/2)
        dxyz = dxyz[nneb+1:]

        if self.dim == 2:
            # keep only in-plane neighbors (at the same 'z' position):
            dxyz = dxyz[dxyz[:,2]==0]
            self.selem = self.selem[self.radius]

        self.dxyz = dxyz
        self.nneb = len(dxyz)
        #logging.info("selem: {}".format(self.selem))
        #logging.info("Relative position of neighbors: {}.".format(self.dxyz))
        #logging.info("Half-number of neighbors to be considered: {}.".format(self.nneb))

    def __str__(self):
        return f"{self.radius}, {self.connectivity}, {self.dim}: {self.nneb}, {self.dxyz}"

    def __repr__(self):
        return (f"{type(self).__name__}"
                f'(radius={self.radius}, '
                f'connectivity="{self.connectivity}", '
                f"dimension={self.dim}, "
                f"half-number of neighbors={self.nneb}, "
                f'relative position of neighbors="{self.dxyz}")')

selem_default = StructuringElement()
selem_default.set_selem(radius=1, connectivity='face', dim=3)

@dataclass
class OriMapParameters:
    filename: str = 'themap.vtk'
    dimensions: Tuple[int, int, int] = (32, 32, 32)
    spacing: Tuple[float, float, float] = (1., 1., 1.)
    origin: Tuple[float, float, float] = (0., 0., 0.)
    phases: List[int] = field(default_factory=lambda: [1])
    phase_to_crys: dict = field(default_factory= lambda: {1: Crystal(1, 'phase1')})
    selem: StructuringElement = field(default_factory=lambda: selem_default)
    thres_HAGB: float = 10.
    thres_LAGB: float = 1.
    grain_cluster_method: str = 'label'
    grain_size_bounds: List[int] = field(default_factory=lambda: [1, 2**31])
    grain_phase_bounds: List[int] = field(default_factory=lambda: [1, 255])
    dim3D: bool = True
    compute_mode: str = 'numba_cpu'
    #grain_size_bounds: Tuple[int, int] = (1, 2**31)
    #grain_phase_bounds: Tuple[int, int] = (1, 255)
    #phases: np.ndarray[(int,), np.dtype[DTYPEi]] = field(default_factory=lambda: np.array([1]))

@dataclass
class Crystal:
    phase: int = 1
    name: str = 'no_name'
    #abc: np.ndarray[(int,), np.dtype[DTYPEf]] = field(default_factory=lambda: np.ones(shape=(3,), dtype=DTYPEf))
    #ang: np.ndarray[(int,), np.dtype[DTYPEf]] = field(default_factory=lambda: np.ones(shape=(3,), dtype=DTYPEf)*90)
    abc: np.ndarray[(int,), np.dtype[DTYPEf]] = field(default_factory=lambda: np.ones(shape=(3,)))
    ang: np.ndarray[(int,), np.dtype[DTYPEf]] = field(default_factory=lambda: np.ones(shape=(3,))*90)
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
            if not self.name == 'unindexed':
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
        elif self.sym == 'none':
            self.qsym = np.array([[1,0,0,0]], dtype=DTYPEf)
        else:
            logging.warning("No quaternion symmetry operator implemented for crystal {self.sym}.")

class OriMap(pv.ImageData):
    """
    OriMap class (inheriting from pyvista ImageData).

    Data is stored at the cell centers (as cell_data) by default rather than at the vertices.

    Parameters
    ----------
    uinput : str, vtk.vtkImageData, pyvista.ImageData, optional
        Filename or dataset to initialize the uniform grid from.
        If set, dimensions, spacing and origin parameters are ignored.
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
    get_grains(params)
        Detect grains based on cell neighborhood.
    """

    def __init__(self, uinput=None, params=OriMapParameters()):
        self.params = params
        super().__init__(uinput,
                         dimensions=self.params.dimensions,
                         spacing=self.params.spacing,
                         origin=self.params.origin)

        # force params update based on the vtk file parameters if directly provided by the user:
        if not uinput is None:
            logging.info("Updating parameters based on vtk file properties.")
            self.params.dimensions = self.dimensions
            self.params.spacing = self.spacing
            self.params.origin = self.origin

        # assessing 2D or 3D:
        if self.params.dimensions[2]-1 > 1: # 3D map
            self.params.dim3D = True
        else: # 2D map, check if dim3D and selem attributes need to be updated:
            if self.params.dim3D:
                self.params.dim3D = False
                # structuring element needs to be simplified for 2D:
                logging.info("Simplifying structuring element for 2D.")
                r = self.params.selem.radius; connec = self.params.selem.connectivity
                self.params.selem.set_selem(r, connec, dim=2)

        # force dtype of points to np.float32? : ImageData.points can't be set...

    def get_IPF_proj(self, proj='stereo'):
        """
        Compute IPF projection.
        """
        sampleAx = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=DTYPEf)

        # Check for the presence of qarray:
        phase = self.cell_data['phase']
        try:
            ncrys = len(self.qarray)
        except AttributeError:
            logging.info("... Starting to generate quaternion array from Euler angles.")
            if self.params.compute_mode == 'numba_gpu' or self.params.compute_mode == 'numba_cpu':
                self.qarray = q4nCPU.q4_from_eul(self.cell_data['eul'])
            else:
                self.qarray = q4np.q4_from_eul(self.cell_data['eul'])
            logging.info("... Finished to generate quaternion array from Euler angles.")
            ncrys = len(self.qarray)

        logging.info("Starting to compute IPF projection.")

        # loop by phase ID for IPF calculation:
        thephases = sorted(list(self.params.phase_to_crys.keys()))
        self.cell_data['IPFx'] = np.zeros((ncrys,3), dtype=DTYPEf)
        self.cell_data['IPFy'] = np.zeros((ncrys,3), dtype=DTYPEf)
        self.cell_data['IPFz'] = np.zeros((ncrys,3), dtype=DTYPEf)
        for phi in thephases:
            if phi == 0: # unindexed
                continue
            logging.info("... phase {}.".format(phi))
            whrPhi = (phase == phi)
            qarr = self.qarray[whrPhi]
            nqPhi = len(qarr)

            try:
                qsym = self.params.phase_to_crys[phi].qsym
            except AttributeError:
                self.params.phase_to_crys[phi].infer_symmetry()
                self.params.phase_to_crys[phi].get_qsym()
                qsym = self.params.phase_to_crys[phi].qsym

            for iax, axis in enumerate(sampleAx):
                logging.info("... ... axis {}.".format(iax))
                if   self.params.compute_mode == 'numba_gpu':
                    iproj = 0 if proj == 'stereo' else 1

                    qarr_gpu = cp.asarray(qarr)
                    qsym_gpu = cp.asarray(qsym)
                    axis_gpu = cp.asarray(axis)
                    xyproj_gpu = cp.zeros((nqPhi,2), dtype=DTYPEf)
                    RGB_gpu = cp.zeros((nqPhi,3), dtype=DTYPEf)
                    albeta_gpu = cp.zeros((nqPhi,2), dtype=DTYPEf)+360
                    isym_gpu = cp.zeros(nqPhi, dtype=np.uint8)
                    q4nGPU.q4_to_IPF(qarr_gpu, axis_gpu, qsym_gpu, iproj, 3, 1, xyproj_gpu, RGB_gpu, albeta_gpu, isym_gpu)

                    xyproj = xyproj_gpu.get()
                    RGB = RGB_gpu.get()
                    albeta = albeta_gpu.get()
                    isym = isym_gpu.get()
                elif self.params.compute_mode == 'numba_cpu':
                    iproj = 0 if proj == 'stereo' else 1
                    xyproj, RGB, albeta, isym = q4nCPU.q4_to_IPF(qarr, axis, qsym, proj=iproj, north=3, method=1)
                else:
                    xyproj, RGB, albeta, isym = q4np.q4_to_IPF(qarr, axis=axis, qsym=qsym, proj=proj, north=3, method=1)

                if iax == 0:
                    self.cell_data['IPFx'][whrPhi] = RGB
                elif iax == 1:
                    self.cell_data['IPFy'][whrPhi] = RGB
                elif iax == 2:
                    self.cell_data['IPFz'][whrPhi] = RGB

        if self.params.compute_mode == 'numba_gpu':
            # free GPU memory for future calculation:
            qarr_gpu =   qarr_gpu[0:1]*0.
            qsym_gpu = qsym_gpu[0:1]*0.
            xyproj_gpu = xyproj_gpu[0:1]*0.
            RGB_gpu = RGB_gpu[0:1]*0.
            albeta_gpu = albeta_gpu[0:1]*0.
            isym_gpu = isym_gpu[0:1]*0.


        logging.info("Finished to compute IPF projection.")

        self.save(self.params.filename[:-4]+'.vtk')

    def get_grains(self):
        """
        Compute grains based on disorientation with cell neighbors.

        Cell neighborhood is defined by a structuring element
        with the input `radius` and `connectivity` as defined in self.params.selem.
        """

        #self._move_to_FZ_by_phase()
        self._get_neighborhood()

        logging.info("Grain cluster method: {}.".format(self.params.grain_cluster_method))
        if self.params.grain_cluster_method == 'graph':
            self._build_cell_graph(symmetric=False)
            self._get_connected_components()
        else:
            self._get_labels()
        self._filter_grains()
        self._get_KAM()
        #self._get_average_grain_by_grain()
        self._get_average_grains_by_phase()
        self.save(self.params.filename[:-4]+'.vtk')

    def _get_KAM(self):
        """
        Compute KAM from neighbor disorientation.
        """
        grains = self.cell_data['grains']
        KAM = np.zeros(self.n_cells, dtype=DTYPEf)
        nKAM = np.zeros(self.n_cells, dtype=np.uint16)
        #self.cell_data['KAM'] = -1 # np.nan

        try:
            shape = self.deso.shape
        except AttributeError:
            logging.error("Neighor disorientation must be calculated beforehand for KAM computation.")

        for ineb in range(self.params.selem.nneb):
            desoNeb = self.deso[:,ineb]
            theNeb = self.neighbors[:,ineb]
            graNeb = grains[theNeb]
            whrSameGr = (grains == graNeb) * (theNeb >= 0)

            KAM[whrSameGr] += desoNeb[whrSameGr]
            nKAM[whrSameGr] += 1
            # also add the opposite neighbor direction :
            KAM[theNeb[whrSameGr]] += desoNeb[whrSameGr]
            nKAM[theNeb[whrSameGr]] += 1

        whrKAM = (nKAM > 0)
        KAM[whrKAM] = KAM[whrKAM]/nKAM[whrKAM]
        KAM[~whrKAM] = np.nan

        self.cell_data['KAM'] = KAM
        logging.info("Finished to compute KAM up to {} neighbors.".format(2*self.params.selem.nneb))

    def _get_cell_coords_from_points(self, method=1):
        """
        Get cell coordinates from the coordinates of cell vertices.

        Note: amounts to removing the last slice/row/col in (gridded) image data.
        """
        points = self.points.astype(DTYPEf)
        if method == 1:
            tol = np.array(self.spacing, dtype=DTYPEf).min()/100.
            xmax = self.bounds[1]-tol; ymax = self.bounds[3]-tol; zmax = self.bounds[5]-tol
            whrC =  (points[:,0] < xmax)*\
                    (points[:,1] < ymax)*\
                    (points[:,2] < zmax)
            return points[whrC]
        else:
            # construct the point mask explicitely for faster access
            pass

    def _get_neighborhood(self, disori=True):
        """
        Get neighbors with given connectivity and corresponding disorientation.
        NB1: disorientation is set to +362. for potential cell neighbors outside of the grid bounds.
        NB2: disorientation is set to +361. for cell neighbors belonging to different phases.
         """
        logging.info("Starting to retrieve neighbors with {} connectivity and radius {}.".format(self.params.selem.connectivity, self.params.selem.radius))

        # get the coordinates of the cells (dimensions - 1 wrt the points):
        xyzC = self._get_cell_coords_from_points(method=1)
        dx, dy, dz = self.spacing
        nn = self.n_cells

        #cellid = np.arange(nn).astype(DTYPEi)
        self.deso = np.zeros((nn,self.params.selem.nneb), dtype=DTYPEf)
        self.neighbors = np.zeros((nn,self.params.selem.nneb), dtype=DTYPEi)

        logging.info("Starting to retrieve neighbors.")
        # loop over the different types of neighbors defined by the structuring element:
        for ineb, dxyz in enumerate(self.params.selem.dxyz):
            logging.info("... ineb {}".format(ineb))
            self.neighbors[:,ineb] = self._get_neighbors_for_ineb(xyzC, dxyz, ineb)

        if disori:
            logging.info("Starting to compute neighbor disorientation.")
            self._get_disori_by_phase()
            logging.info("Finished to compute neighbor disorientation.")

    def _get_neighbors_for_ineb(self, xyzC, dxyz, ineb):
        """
        Construct cell neighborhood information for the ineb_th family of neighbors.
        """

        dz = dxyz[2]; dy = dxyz[1];  dx = dxyz[0]
        dz *= self.spacing[2]; dy *= self.spacing[1];  dx *= self.spacing[0]
        dz = dz.astype(DTYPEf); dy = dy.astype(DTYPEf); dx = dx.astype(DTYPEf)
        z = xyzC[:,2]; y = xyzC[:,1]; x = xyzC[:,0]

        dimensions = np.array(list(self.dimensions), dtype=DTYPEi)
        spacing = np.array(list(self.spacing), dtype=DTYPEf)
        bounds = np.array(list(self.bounds), dtype=DTYPEf)
        if (self.params.compute_mode == 'numba_cpu') or (self.params.compute_mode == 'numba_gpu') or (self.params.compute_mode == 'cupy'):
            ii = xyz_to_index_numba(x+dx, y+dy, z+dz, dimensions, spacing, bounds, mode=1)
        else:
            ii = xyz_to_index(x+dx, y+dy, z+dz, dimensions, spacing, bounds, mode='cells')
        return ii

    def _move_to_FZ_by_phase(self):
        """
        Move quaternions to Fundamental Zone, phase by phase.
        """
        logging.info("Starting to move quaternions to Fundamental Zone.")
        phase = self.cell_data['phase']
        try:
            ncrys = len(self.qarray)
        except AttributeError:
            logging.info("... Starting to generate quaternion array from Euler angles.")
            if self.params.compute_mode == 'numba_gpu' or self.params.compute_mode == 'numba_cpu':
                self.qarray = q4nCPU.q4_from_eul(self.cell_data['eul'])
            else:
                self.qarray = q4np.q4_from_eul(self.cell_data['eul'])
            logging.info("... Finished to generate quaternion array from Euler angles.")

        # loop by phase ID for disorientation calculation:
        thephases = sorted(list(self.params.phase_to_crys.keys()))
        for phi in thephases:
            if phi == 0: # unindexed
                continue
            logging.info("... phase {}.".format(phi))
            whrPhi = (phase == phi)
            qarr = self.qarray[whrPhi]

            try:
                qsym = self.params.phase_to_crys[phi].qsym
            except AttributeError:
                self.params.phase_to_crys[phi].infer_symmetry()
                self.params.phase_to_crys[phi].get_qsym()
                qsym = self.params.phase_to_crys[phi].qsym

            if self.params.compute_mode == 'numba_gpu':
                qarr_gpu = cp.asarray(qarr)
                qsym_gpu = cp.asarray(qsym)
                qFZ_gpu = cp.zeros_like(qarr_gpu)
                q4nGPU.q4_to_FZ(qarr_gpu, qsym_gpu, qFZ_gpu)
                self.qarray[whrPhi] = cp.asnumpy(qFZ_gpu)

                qarr_gpu = qarr_gpu[0:1]*0.
                qFZ_gpu = qarr_gpu[0:1]*0.
                mempool.free_all_blocks()
            elif self.params.compute_mode == 'numba_cpu':
                self.qarray[whrPhi], _ = q4nCPU.q4_to_FZ(qarr, qsym)
            else:
                self.qarray[whrPhi] = q4np.q4_to_FZ(qarr, qsym)
        logging.info("Finished to move quaternions to Fundamental Zone.")

    def _get_disori_by_phase(self):
        """
        Compute the disorientation for all families of neighbors, one phase after the other.

        NB1: disorientation is set to +362. for potential cell neighbors outside of the grid bounds.
        NB2: disorientation is set to +361. for cell neighbors belonging to different phases.
        """
        logging.info("... compute_mode for disorientation: {}".format(self.params.compute_mode))
        phase = self.cell_data['phase']
        try:
            ncrys = len(self.qarray)
        except AttributeError:
            logging.info("... Starting to generate quaternion array from Euler angles.")
            if self.params.compute_mode == 'numba_gpu' or self.params.compute_mode == 'numba_cpu':
                self.qarray = q4nCPU.q4_from_eul(self.cell_data['eul'])
            else:
                self.qarray = q4np.q4_from_eul(self.cell_data['eul'])
            logging.info("... Finished to generate quaternion array from Euler angles.")

        if (self.params.compute_mode == 'cupy') or (self.params.compute_mode == 'numba_gpu'):
            qarr_gpu = cp.asarray(self.qarray)

        # loop by phase ID for disorientation calculation:
        thephases = sorted(list(self.params.phase_to_crys.keys()))
        for phi in thephases:
            whrPhi = (phase == phi)
            neighb = self.neighbors[whrPhi]
            qa = self.qarray[whrPhi]

            if phi == 0: # unindexed
                for ineb in range(self.params.selem.nneb):
                    theNeb = neighb[:,ineb]
                    whrNeb = (theNeb >= 0) # where False, the neighbor is out of the grid bounds
                    phaseNeb = phase[theNeb]
                    whrPhiNeb = (phaseNeb == phi) # where False, the neighbor belongs to another phase

                    self.deso[whrPhi][:,ineb] = -1.
                continue

            try:
                qsym = self.params.phase_to_crys[phi].qsym
            except AttributeError:
                self.params.phase_to_crys[phi].infer_symmetry()
                self.params.phase_to_crys[phi].get_qsym()
                qsym = self.params.phase_to_crys[phi].qsym

            if (self.params.compute_mode == 'numba_gpu'):
                qsym_gpu = cp.asarray(qsym)
                qa_gpu = qarr_gpu[whrPhi]
                deso_gpu = cp.asarray(self.deso[whrPhi])
            elif (self.params.compute_mode == 'cupy'):
                qsym_gpu = cp.asarray(qsym)
                qa_gpu = qarr_gpu[whrPhi]
                deso_gpu = cp.asarray(self.deso[whrPhi])
                qc_gpu = cp.zeros_like(qa_gpu)
                a1_gpu = cp.zeros_like(deso_gpu[:,0])
                print('GPU mem, start, used:', mempool.used_bytes()/1024**2)
                print('GPU mem, start, total:', mempool.total_bytes()/1024**2)
            else:
                deso = self.deso[whrPhi]

            # loop by neighbor type:
            for ineb in range(self.params.selem.nneb):
                logging.info("... phase {}, ineb {}".format(phi, ineb))
                theNeb = neighb[:,ineb]

                if self.params.compute_mode == 'numba_cpu':
                    qb = self.qarray[theNeb]
                    deso[:,ineb], _ = q4nCPU.q4_disori_angle(qa, qb, qsym, method=1)
                elif self.params.compute_mode == 'numba_gpu':
                    qb_gpu = qarr_gpu[theNeb]
                    q4nGPU.q4_disori_angle(qa_gpu, qb_gpu, qsym_gpu, deso_gpu[:,ineb], nthreads=256)
                    qb_gpu = qb_gpu[0:1]*0. # free memory for next neighbor
                elif self.params.compute_mode == 'cupy':
                    qb_gpu = qarr_gpu[theNeb]
                    q4cp.q4_disori_angle(qa_gpu, qb_gpu, qc_gpu, qsym_gpu,
                                         deso_gpu[:,ineb], a1_gpu, method=1, revertqa=True)
                    print('GPU mem, middle, used:', mempool.used_bytes()/1024**2)
                    print('GPU mem, middle, total:', mempool.total_bytes()/1024**2)
                    qb_gpu = qb_gpu[0:1]*0. # free memory for next neighbor
                else: # default to numpy mode:
                    qb = self.qarray[theNeb]
                    deso[:,ineb] = q4np.q4_disori_angle(qa, qb, qsym, method=1, return_index=False)

            # transfer results and reset gpu_arrays before jumping to next phase if cupy or numba_gpu:
            if (self.params.compute_mode == 'cupy') or (self.params.compute_mode == 'numba_gpu'):
                self.deso[whrPhi] = cp.asnumpy(deso_gpu)

                # free memory for next phase:
                qsym_gpu = qsym_gpu[0:1]*0.
                qa_gpu =   qa_gpu[0:1]*0.
                deso_gpu = deso_gpu[0:1]*0.
                if (self.params.compute_mode == 'cupy'):
                    qc_gpu =   qc_gpu[0:1]*0.
                    a1_gpu = a1_gpu[0:1]
            else:
                self.deso[whrPhi] = deso

        # final pass on the neighbors to "erase" unproper neighbor disorientations:
        logging.info("... updating unproper neighbor disorientations...")
        for ineb in range(self.params.selem.nneb):
            theNeb = self.neighbors[:,ineb]
            whrNeb = (theNeb >= 0)
            phiNeb = phase[theNeb]
            whrPhi = (phase == phiNeb)
            self.deso[:,ineb][~whrPhi] = 361.
            self.deso[:,ineb][~whrNeb] = 362.
            logging.info("... ineb {}".format(ineb))

        if (self.params.compute_mode == 'cupy') or (self.params.compute_mode == 'numba_gpu'):
            qarr_gpu = qarr_gpu[0:1]*0.
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            #print('GPU mem, final, used:', mempool.used_bytes()/1024**2)
            #print('GPU mem, final, total:', mempool.total_bytes()/1024**2)

    def _get_connected_components(self):
        """
        Get the connected components based on given threshold.
        """
        ncomp, labels = sparse.csgraph.connected_components(self.Adjmat, directed=False, return_labels=True)

        self.cell_data['region'] = labels + 1 # starting at 1 instead of 0

        logging.info("Finished to compute {} connected components.".format(ncomp))

    def _build_cell_graph(self, symmetric=False):
        """
        Build the graph of cell connections as a scipy.sparse csr_array.
        """
        logging.info("Starting to build cell graph with {} deg threshold for HAGB.".format(self.params.thres_HAGB))

        nn = self.n_cells
        cellid = np.arange(nn).astype(DTYPEi)

        whr = np.where((self.neighbors >= 0)*(self.deso < self.params.thres_HAGB))
        self.Adjmat = sparse.csr_array((self.deso[whr], (cellid[whr[0]], self.neighbors[whr])),
                                       shape=(nn,nn), dtype=DTYPEf)
        if symmetric:
            self.Adjmat += self.Adjmat.T

        logging.info("Finished to build cell graph.")

    def _get_labels(self):
        """
        Get labeled regions based on neighborhood and disorientation values.
        """
        logging.info("Starting to get labels with {} deg threshold for HAGB.".format(self.params.thres_HAGB))

        labels, tocorr = loop_on_cells_neighbors(self.neighbors, self.deso, thres=self.params.thres_HAGB, nlabmax=8*2**16, ncorrmax=256)

        reg = np.repeat(np.arange(tocorr.shape[0]), tocorr.shape[1])
        nebreg = tocorr.flatten()
        whr = (nebreg > 0)
        reg = reg[whr]
        nebreg = nebreg[whr] - 1
        val = np.ones(nebreg.size, dtype=np.uint8)

        Adjmat = sparse.csr_array((val, (reg, nebreg)),
                                shape=(tocorr.shape[0], tocorr.shape[0]), dtype=np.uint8)
        ncomp, newlab = sparse.csgraph.connected_components(Adjmat, directed=False, return_labels=True)

        unic, iback, counts = np.unique(labels, return_inverse=True, return_counts=True)

        self.cell_data['region'] = newlab[iback] + 1 # starting at 1 instead of 0

        logging.info("Finished to compute {} region labels.".format(ncomp))

    def _filter_grains(self):
        """
        Relabel grains in a consecutive sequence after the exclusion of regions outside of ncell_range and phase_range.
        """

        # why not using ndimage.label directly for this relabeling operation?
        nmin = self.params.grain_size_bounds[0]; nmax = self.params.grain_size_bounds[1]
        phimin = self.params.grain_phase_bounds[0]; phimax = self.params.grain_phase_bounds[1]
        phase = self.cell_data['phase']
        region = self.cell_data['region']


        filtered = False
        # first, exclusion of regions outside of phase_min and phase_max:
        if (phase.min() < phimin) or (phase.max() > phimax):
            filtered = True
            whr = (phase < phimin) + (phase >= phimax)
            region[whr] = 0
        unic, iback, counts = np.unique(region, return_inverse=True, return_counts=True)

        # then exclusion of regions outside of npix_min and npix_max:
        if (counts.min() < nmin) or (counts.max() > nmax):
            filtered = True
            toremove = (counts < nmin) + (counts >= nmax)
            unic[toremove] = 0
            region = unic[iback]
            # re-run the counting:
            unic, iback, counts = np.unique(region, return_inverse=True, return_counts=True)

        # redefine the unique grain label in a consecutive sequence:
        if filtered:
            newu = np.arange(len(unic))
            if unic.min() == 0:
                self.cell_data['grains'] = newu[iback] # preserve '0' region with label 0
            elif unic.min() >= 1:
                self.cell_data['grains'] = newu[iback]+1 # there is no region outside of the specified ranges, start labeling at 1
            # final count:
            unic, counts = np.unique(self.cell_data['grains'], return_counts=True)

            logging.info("Finished to relabel {} grains with min size {} and min phase {}.".format(unic.max(), nmin, phimin))
        else:
            self.cell_data['grains'] = region
            logging.info("No grain to relabel with min size {} and min phase {}.".format(nmin, phimin))


        return unic, counts

    def _get_average_grains_by_phase(self):
        """
        Compute average orientation, GOS and GROD for all grains at once in each phase.
        """
        grains = self.cell_data['grains']
        phase = self.cell_data['phase']
        KAM = self.cell_data['KAM']
        GROD = np.zeros(grains.shape, dtype=DTYPEf)
        GOS = np.zeros(grains.shape, dtype=DTYPEf)
        theta = np.zeros(grains.shape, dtype=DTYPEf)
        qavg = np.zeros_like(self.qarray)

        logging.info("Starting to compute grain average orientation and GROD (multigrain).")

        thephases = sorted(list(self.params.phase_to_crys.keys()))
        for phi in thephases:
            if phi > 0: # indexed phases
                logging.info("... phase {}:".format(phi))
                whrPhi = (phase == phi)*(grains > 0)
                if np.sum(whrPhi) == 0:
                    logging.info("No pixel is this phase anymore using present grain filtering parameters. Skipping it for grain averages.")
                    continue

                qaPhi = self.qarray[whrPhi]
                qsym = self.params.phase_to_crys[phi].qsym
                grPhi = grains[whrPhi]

                if (self.params.compute_mode == 'numba_gpu'):
                    gr_gpu = cp.asarray(grPhi, dtype=DTYPEi)

                    u, iu, ib = cp.unique(gr_gpu, return_index=True, return_inverse=True)
                    # articial transfer to CPU memory to get rid of (huge amount of) cached GPU memory during cp.unique()...:
                    unic = u.astype(DTYPEi).get(); iunic = iu.astype(DTYPEi).get(); iback = ib.astype(DTYPEi).get()
                    u = None; iu = None; ib = None
                    mempool.free_all_blocks()

                    qa_gpu = cp.asarray(qaPhi, dtype=DTYPEf)
                    qsym_gpu = cp.asarray(qsym, dtype=DTYPEf)
                    qavg_gpu, GOS_gpu, theta_gpu, GROD_gpu, theta_iter_gpu = q4nGPU.q4_mean_multigrain(qa_gpu,
                                                                                                       qsym_gpu,
                                                                                                       cp.asarray(unic),
                                                                                                       cp.asarray(iunic),
                                                                                                       cp.asarray(iback))
                    qavgPhi = qavg_gpu.get()
                    GOSPhi = GOS_gpu.get()
                    thetaPhi = theta_gpu.get()
                    GRODPhi = GROD_gpu.get()
                    theta_iter = theta_iter_gpu.get()
                elif (self.params.compute_mode == 'numba_cpu'):
                    unic, iunic, iback = np.unique(grPhi, return_index=True, return_inverse=True)
                    qavgPhi, GOSPhi, thetaPhi, GRODPhi, theta_iter = q4nCPU.q4_mean_multigrain(qaPhi, qsym, unic, iunic, iback)
                else:
                    unic, iunic, iback = np.unique(grPhi, return_index=True, return_inverse=True)
                    qavgPhi, GOSPhi, thetaPhi, GRODPhi, theta_iter = q4np.q4_mean_multigrain(qaPhi, qsym, unic, iunic, iback)

                qavg[whrPhi] = qavgPhi[iback]
                GOS[whrPhi] = GOSPhi[iback]
                theta[whrPhi] = thetaPhi[iback]
                GROD[whrPhi] = GRODPhi

        self.cell_data['GROD'] = GROD
        self.cell_data['GOS'] = GOS
        self.cell_data['theta'] = theta

        if self.params.compute_mode == 'numba_gpu':
            # free GPU memory for future calculation:
            qa_gpu =   qa_gpu[0:1]*0.
            qsym_gpu = qsym_gpu[0:1]*0.
            qavg_gpu = qavg_gpu[0:1]*0.
            GOS_gpu = GOS_gpu[0:1]*0.
            theta_gpu = theta_gpu[0:1]*0.
            GROD_gpu = GROD_gpu[0:1]*0.
            theta_iter_gpu = theta_iter_gpu[0:1]*0.

        # save average grain data:
        mydtype = [('grain', 'i4'), ('phase', 'i4'), ('npix', 'i4'), ('fvol', 'f4'),
                   ('phi1', 'f4'), ('Phi', 'f4'), ('phi2', 'f4'), ('GOS', 'f4'), ('KAMavg', 'f4'), ('theta', 'f4')]
        unic, iunic, counts = np.unique(grains, return_index=True, return_counts=True)
        grdata = np.rec.array(np.zeros(len(unic), dtype=mydtype))

        qavg1 = qavg[iunic]
        eul = q4np.q4_to_eul(qavg1)
        grdata.grain = unic
        grdata.phase = phase[iunic]
        grdata.npix = counts
        grdata.fvol = counts.astype(np.float32)/np.sum(counts)
        grdata.phi1 = eul[:,0]
        grdata.Phi = eul[:,1]
        grdata.phi2 = eul[:,2]
        grdata.GOS = GOS[iunic]
        grdata.theta = theta[iunic]
        grdata.KAMavg = ndi.mean(KAM, labels=grains, index=unic)
        np.savetxt(self.params.filename[:-4]+'-grains.txt', grdata,
                fmt="%6i %4i %9i %10.6f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f", header=str(grdata.dtype.names))

        logging.info("Finished to compute grain average orientation and GROD (multigrain).")

    def _get_average_grain_by_grain(self):
        """
        Compute average orientation, GOS and GROD, grain by grain.
        """
        grains = self.cell_data['grains']
        phase = self.cell_data['phase']
        GROD = np.zeros(grains.shape, dtype=DTYPEf)
        GOS = np.zeros(grains.shape, dtype=DTYPEf)

        logging.info("Starting to compute grain average orientation and GROD (grain by grain).")

        unic, counts = np.unique(grains, return_counts=True)
        labcount = np.column_stack((unic, counts, np.cumsum(counts)))
        labcount = np.rec.array( unstructured_to_structured(labcount, dtype=[('lab', DTYPEi), ('count', DTYPEi), ('cum', DTYPEi)]) )

        grind = np.column_stack((grains, np.arange(grains.size)))
        grind = np.rec.array( unstructured_to_structured(grind, dtype=[('lab', DTYPEi), ('index', DTYPEi)]) )
        grind.sort()

        for ilab, lab in enumerate(labcount.lab):
            if ilab == 0:
                indices = grind.index[0:labcount.cum[ilab]]
            else:
                indices = grind.index[labcount.cum[ilab-1]:labcount.cum[ilab]]
            #print("lab, indices:", lab, indices)

            if lab == 0:
                GROD[indices] = 0.
                GOS[indices] = 0.
                continue

            qlab = self.qarray[indices]

            phi = phase[indices][0]
            qsym = self.params.phase_to_crys[phi].qsym
            if (self.params.compute_mode == 'numba_gpu'):
                qa_gpu = cp.asarray(qlab, dtype=DTYPEf)
                qsym_gpu = cp.asarray(qsym, dtype=DTYPEf)
                qavg_gpu = cp.zeros((1,4), dtype=DTYPEf)
                GROD_gpu = cp.zeros(len(qlab), dtype=DTYPEf)
                GROD_stat_gpu = cp.zeros(7, dtype=DTYPEf)
                theta_iter_gpu = cp.zeros(10, dtype=DTYPEf)
                q4nGPU.q4_mean_disori(qa_gpu, qsym_gpu, qavg_gpu, GROD_gpu, GROD_stat_gpu, theta_iter_gpu)

                qavg = cp.asnumpy(qavg_gpu)
                GROD[indices] = cp.asnumpy(GROD_gpu)
                GROD_stat = cp.asnumpy(GROD_stat_gpu)
                theta_iter = cp.asnumpy(theta_iter_gpu)
            elif (self.params.compute_mode == 'numba_cpu'):
                qavg, GROD[indices], GROD_stat, theta_iter = q4nCPU.q4_mean_disori(qlab, qsym)
            else:
                qavg, GROD[indices], GROD_stat, theta_iter = q4np.q4_mean_disori(qlab, qsym)
            GOS[indices] = GROD_stat[0]

        self.cell_data['GROD'] = GROD
        self.cell_data['GOS'] = GOS

        if self.params.compute_mode == 'numba_gpu':
            # free GPU memory for future calculation:
            qa_gpu =   qa_gpu[0:1]*0.
            qsym_gpu = qsym_gpu[0:1]*0.
            qavg_gpu = qavg_gpu[0:1]*0.
            GROD_gpu = GROD_gpu[0:1]*0.
            GROD_stat_gpu = GROD_stat_gpu[0:1]*0.
            theta_iter_gpu = theta_iter_gpu[0:1]*0.

        logging.info("Finished to compute grain average orientation and GROD (grain by grain).")

    def _save_phase_info(self):
        """
        Save the properties of individual phases and crystals to .phi file.
        """
        try:
            phases = list(self.params.phase_to_crys.keys())
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
            thephase = self.params.phase_to_crys[phi]
            crys.append( (thephase.name, thephase.phase, thephase.sym,
                          thephase.abc[0], thephase.abc[1], thephase.abc[2],
                          thephase.ang[0], thephase.ang[1], thephase.ang[2]) )
        crys = np.array(crys, dtype=mydtype)
        filename = self.params.filename[:-4]+'.phi'
        np.savetxt(filename, crys, delimiter=',', fmt=fmt, header=str(crys.dtype.names)[1:-1])

    def _read_phase_info(self, f=None):
        """
        Read the properties of individual phases and crystals from .phi file.
        """
        if f is None:
            f = self.params.filename[:-4]+'.phi'

        nophasedata = True
        try:
            thephases = np.atleast_1d( np.unique(self.cell_data['phase']) )
            nophasedata = False
        except KeyError:
            logging.warning("Phase data wasn't found within available cell_data keys: {}".format(self.cell_data.keys()))
            nophasedata = True

        if nophasedata:
            logging.warning("Assuming homogeneous 'phase1' with cubic symmetry for the whole microstructure.")
            self.params.phase_to_crys[1] = Crystal(1, name='phase1', sym='cubic')
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
                self.params.phases = list(phases)
                for phi in phases:
                    self.params.phase_to_crys[phi.phase] = Crystal(phi.phase, name=phi.name, sym=phi.sym,
                                                            abc=np.array([phi.a,phi.b,phi.c,]),
                                                            ang=np.array([phi.alpha,phi.beta,phi.gamma,]))
            else:
                logging.warning("Assuming cubic symmetry for all phases ({}).".format(thephases))
                self.params.phases = list(thephases)
                for phi in thephases:
                    self.params.phase_to_crys[phi] = Crystal(phi, name='phase'+str(phi), sym='cubic')

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

    selem = StructuringElement()
    selem.set_selem(radius=1, connectivity='face', dim=2)
    params = OriMapParameters(filename=filename,
                              dimensions=(XCells+1, YCells+1, 2),
                              spacing=(XStep, YStep, max(XStep, YStep)),
                              phases=list(phase_to_crys.keys()),
                              phase_to_crys=phase_to_crys,
                              selem=selem,
                              dim3D=False)
    params.JobMode = JobMode
    params.AcqE1 = AcqE1
    params.AcqE2 = AcqE2
    params.AcqE3 = AcqE3
    params.EulerFrame = EulerFrame

    orimap = OriMap(None, params)

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
    orimap._save_phase_info()

    return orimap

def read_from_vtk(filename):
    """
    Read a .vtk file and return an OriMap object.
    """
    logging.info("Reading data from {}.".format(filename))

    params = OriMapParameters(filename=filename)

    orimap = OriMap(uinput=filename, params=params)
    orimap._read_phase_info()

    phases = np.unique(orimap.cell_data['phase'])
    # crystal definition:
    propercrys = True
    keys = np.sort( np.array( list(orimap.params.phase_to_crys.keys()) ) )
    if not np.allclose(phases, keys):
        propercrys = False
        logging.warning("Crystal dictionary not specified properly. Default crystals will be attributed to the phases.")

    if not propercrys:
        for iphase, phase in enumerate(phases):
            orimap.params.phase_to_crys[phase] = Crystal(phase, name='phase'+str(phase))
            logging.info("{}".format(orimap.params.phase_to_crys[phase]))
    else:
        for iphase, phase in enumerate(phases):
            logging.info("{}".format(orimap.params.phase_to_crys[phase]))

    logging.info("Finished reading from {}.".format(filename))

    return orimap

def xyz_to_index(x, y, z, dimensions, spacing, bounds, mode='cells', method=1):
    """
    Return the cell/point indices in the grid from (x,y,z) coordinates.
    """
    if mode == 'cells':
        nx, ny, nz = dimensions[0]-1, dimensions[1]-1, dimensions[2]-1
    elif mode == 'points':
        nx, ny, nz = dimensions[0], dimensions[1], dimensions[2]
    dx, dy, dz = spacing
    ox, oy, oz = bounds[0], bounds[2], bounds[4]

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

def index_to_xyz(ii, dimensions, spacing, bounds, mode='cells', method=1):
    """
    Return (x,y,z) coordinates in the grid from cell/point indices.
    """
    if mode == 'cells':
        nx, ny, nz = dimensions[0]-1, dimensions[1]-1, dimensions[2]-1
    elif mode == 'points':
        nx, ny, nz = dimensions[0], dimensions[1], dimensions[2]
    dx, dy, dz = spacing
    ox, oy, oz = bounds[0], bounds[2], bounds[4]

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

@njit(int32[:](float32[:],float32[:],float32[:],int32[:],float32[:],float32[:],int8), fastmath=True, parallel=True)
def xyz_to_index_numba(x, y, z, dimensions, spacing, bounds, mode=1):
    """
    Return the cell/point indices in the grid from (x,y,z) coordinates.
    Numba_cpu version.
    """
    if mode == 1: # cell mode
        nx, ny, nz = dimensions[0]-1, dimensions[1]-1, dimensions[2]-1
    elif mode == 2: # point mode
        nx, ny, nz = dimensions[0], dimensions[1], dimensions[2]
    dx, dy, dz = spacing
    ox, oy, oz = bounds[0], bounds[2], bounds[4]

    nn = len(x)
    ii = np.zeros(nn, dtype=np.int32)
    for i in prange(nn):
        a = np.round((x[i]-ox)/dx);
        b = np.round((y[i]-oy)/dy)
        c = np.round((z[i]-oz)/dz)
        if (a >= 0) and (a < nx) and (b >= 0) and (b < ny) and (c >= 0) and (c < nz):
            c = c*nx*ny
            b = b*nx
            ii[i] = np.int32(c + b + a)
        else:
            ii[i] = -1
    return ii

@njit(nTuple((int32[:], int32[:,:]))(int32[:,:], float32[:,:], float32, int32, int32), fastmath=True)
def loop_on_cells_neighbors(neighbors, deso, thres=5., nlabmax=2**16, ncorrmax=32):
    """
    Loop on cells to define homogeneous regions.
    Numba_cpu version.

    Parameters
    ----------
    neighbors : ndarray
        (ncells, nneb) array of int32 neighbor indices.
    deso : ndarray
        (ncells, nneb) array of float32 disorientation values.
    thres : float
        threshold value below which two neighbors will be connected.
    nlabmax : int, default=2**16
        up to nlabmax different regions (65535 by default).
    ncorrmax : int, default=32
        one region can be merged with up to ncorrmax other regions after the first pass on cell neighbors (32 by default).

    Returns
    -------
    labels : ndarray
        (ncells, ) array of grain region numbers.
    tocorr : ndarray
        (ncorrmax, 2) array of region labels that have to be merged.
    """
    labels = np.zeros_like(neighbors[:,0])
    labmax = 0
    nneb = neighbors.shape[1]
    tocorr = np.zeros((nlabmax,ncorrmax), dtype=np.int32)
    ncorr = np.zeros(nlabmax, dtype=np.int32)
    for icell in range(labels.size):
        if icell % 1000000 == 0:
            print("icell", icell)

        lab = labels[icell]
        if lab == 0:
            neighb = neighbors[icell, :]
            neighbdeso = deso[icell, :]
            neighblab = labels[neighb]
            whr = (neighb >= 0) * (neighbdeso < thres)

            newlab = False
            if whr.sum() > 0:
                mx = neighblab[whr].max()
                if mx > 0:
                    lab = mx
                else:
                    newlab = True
            else:
                newlab = True
            if newlab:
                lab = labels.max() + 1
                if lab >= nlabmax:
                    print("!!!!!!!! Z !!!!!!!! Tentative number of labels exceeding the nlabmax limit", lab, nlabmax)
                    #logging.warning("Tentative number of labels exceeding the nlabmax limit: {} (max {})".format(lab, nlabmax))
                tocorr[lab, 0] = lab
            labels[icell] = lab

        # propagate label to all neighbors
        for ineb in range(nneb):
            theneb = neighbors[icell, ineb]
            if (theneb >= 0) and (deso[icell, ineb] < thres):
                if (labels[theneb] != lab) and (labels[theneb] > 0):
                    lab1 = min([labels[theneb], lab])
                    lab2 = max([labels[theneb], lab])
                    whr = (tocorr[lab1,:] > 0)
                    ncorr = np.sum( whr )
                    if ncorr >= ncorrmax:
                        print("!!!!!!!! Z !!!!!!!! Tentative number of merges exceeding the ncorrmax limit for label", lab, ncorr, ncorrmax)
                        #logging.warning("Tentative number of merges exceeding the ncorrmax limit for label {}: {} (max {})".format(lab, ncorr, ncorrmax))

                    whr = (tocorr[lab1,:] == lab2)
                    if np.sum(whr) == 0:
                        tocorr[lab1, ncorr] = lab2
                    lab = lab1
                labels[theneb] = lab

    col = tocorr.sum(axis=0)
    cmax = np.where(col > 0)[0].max()
    tocorr = tocorr[:,:cmax+1]

    row = tocorr.sum(axis=1)
    rmax = np.where(row > 0)[0].max()
    tocorr = tocorr[:rmax+1,:]

    return labels, tocorr[1:,:]
