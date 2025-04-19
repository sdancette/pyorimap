# -*- coding: utf-8 -*-

# pyorimap/virtual_micro.py

"""
Generate virtual microstructures

The module contains the following functions:

- `Voronoi_microstructure(dimensions)` - generate a Voronoi microstructure
"""

import logging
import numpy as np
import pyvista as pv
import scipy.ndimage as ndi
from scipy.spatial import KDTree

from pyorimap.quaternions import quaternions_np as q4np
from pyorimap.orimap import orientation_map as om

DTYPEf = np.float32
DTYPEi = np.int32

def Voronoi_microstructure(dimensions=(128,128,128), spacing=1, ngrains=5**3, phases=1, fvol=None, phase_to_crys=None, theta_spread=None, spread_type='radial'):
    """
    Generate a random Voronoi microstructure and return an OriMap object
    (inheriting from pyvista ImageData object).

    Data is stored at the cell centers, so that the actual dimensions of
    the ImageData object (which are related to the grid points by default,
    i.e. cell corners) will be (dimX+1, dimY+1, dimZ+1).

    Parameters
    ----------
    dimensions : array_like
        dimensions in the X, Y, Z, directions.
    spacing : float
        size of each (cubic) grid point.
    ngrains : int
        number of grains.
    phases : int or array_like
        phase labels.
    fvol : dict or None, default=None
        dictionary of volume fractions for each phase label (if multiphase).
    phase_to_crys : dict or None, default=None
        dictionary of crystal definition for each phase label (if multiphase).

    Returns
    -------
    grid : pyorimap.OriMap
        OriMap object describing the microstructure.

    Notes
    -----
    OriMap inherits from ImageData, see details of pyvista.ImageData
    at <https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.imagedata>.

    Examples
    --------
    >>> mic = Voronoi_microstructure(dimensions=(64,64,64), spacing=1, ngrains=5**3, theta_spread=1.)
    >>> mic = Voronoi_microstructure(dimensions=(64,64,64), spacing=1, ngrains=5**3, phases=[0,1,2,4], fvol={0:0.15, 1:0.4, 2:0.4, 4:0.05})
    >>> mic = Voronoi_microstructure(dimensions=(64,64,64), spacing=1, ngrains=5**3, phases=5, theta_spread=5.)
    """
    logging.info("#### Starting to build virtual microstructure. ####")

    dimX = dimensions[0]; dimY = dimensions[1]; dimZ = dimensions[2]
    phases = np.atleast_1d(phases).astype(np.uint8)
    phases.sort()
    nphases = len(phases)

    # check consistent phase input:
    if nphases > 1:
        # volume fraction:
        homogeneous = True
        if isinstance(fvol, dict):
            keys = np.sort( np.array( list(fvol.keys()) ) )
            sumval = np.array( list(fvol.values()) ).sum()
            if np.allclose(phases, keys) and (abs(sumval-1.)<0.000001):
                homogeneous = False
            else:
                logging.warning("Volume fraction dictionary not specified properly. Assuming homogeneous distribution of phases.")
        else:
            logging.warning("Volume fraction not specified as a dictionary. Assuming homogeneous distribution of phases.")
        if homogeneous:
            f = 1.0/nphases
            fvol = dict()
            for phase in phases:
                fvol[phase] = f
            logging.warning("Assumed volume fraction: {}".format(fvol))

    # check crystal definition:
    propercrys = True
    if isinstance(phase_to_crys, dict):
        keys = np.sort( np.array( list(phase_to_crys.keys()) ) )
        if not np.allclose(phases, keys):
            propercrys = False
            logging.warning("Crystal dictionary not specified properly. Default crystals will be attributed to the phases.")
    else:
        propercrys = False
        logging.warning("Crystal dictionary not specified. Default crystals will be attributed to the phases.")

    if not propercrys:
        phase_to_crys = dict()
        for iphase, phase in enumerate(phases):
            phase_to_crys[phase] = om.Crystal(phase, name='phase'+str(phase))
            phase_to_crys[phase].infer_symmetry()
            phase_to_crys[phase].get_qsym()
            logging.info("{}".format(phase_to_crys[phase]))
    else:
        for iphase, phase in enumerate(phases):
            logging.info("{}".format(phase_to_crys[phase]))

    # pyvista Image object:
    filename = 'voro-{}-{}-{}.vtk'.format(dimX,dimY,dimZ)
    selem = om.StructuringElement()
    selem.set_selem(radius=1, connectivity='face', dim=3)
    params = om.OriMapParameters(filename=filename,
                              dimensions=(dimX+1, dimY+1, dimZ+1),
                              spacing=(spacing, spacing, spacing),
                              phases=list(phase_to_crys.keys()),
                              phase_to_crys=phase_to_crys,
                              selem=selem)

    grid = om.OriMap(None, params)

    # coordinates of cell centers (starting at zero):
    tol = np.array(grid.spacing, dtype=DTYPEf).min()/100.
    whr = (grid.points[:,0] < grid.bounds[1]-tol)*\
          (grid.points[:,1] < grid.bounds[3]-tol)*\
          (grid.points[:,2] < grid.bounds[5]-tol)
    xyzCells = grid.points[whr].astype(DTYPEf)
    if not grid.n_cells == len(xyzCells):
        logging.error("n_cells does not match with the number of calculated cell centers: {}, {}.".format(grid.n_cells, len(xyzCells)))

    #seeds = np.random.rand(ngrains,3).astype(DTYPEf)
    if grid.params.dim3D:
        nx = (ngrains)**(1./3)
        step = 1./nx
        a = np.arange(0.,1.,step)
        xv, yv, zv = np.meshgrid(a, a, a, sparse=False, indexing='xy')
        seeds = np.zeros((len(xv.flatten()),3), dtype=DTYPEf)
        seeds[:,0] = xv.flatten() + step/2
        seeds[:,1] = yv.flatten() + step/2
        seeds[:,2] = zv.flatten() + step/2
    else:
        nx = np.sqrt(ngrains)
        step = 1./nx
        a = np.arange(0.,1.,step)
        xv, yv = np.meshgrid(a, a, sparse=False, indexing='xy')
        seeds = np.zeros((len(xv.flatten()),3), dtype=DTYPEf)
        seeds[:,0] = xv.flatten() + step/2
        seeds[:,1] = yv.flatten() + step/2
    seeds += (np.random.rand(len(seeds),3)*2 -1)*step/3
    seeds[:,0] *= dimX
    seeds[:,1] *= dimY
    seeds[:,2] *= dimZ
    # shuffle seed position:
    np.random.shuffle(seeds) # shuffled along the 1rst axis
    ngrains = len(seeds)

    logging.info("Starting to compute KDTree.")
    tree = KDTree(seeds)
    dist, grains = tree.query(xyzCells)
    grid.cell_data['grain'] = grains.astype(DTYPEi) + 1
    logging.info("Finished to compute KDTree.")

    grid.cell_data['phase'] = np.ones(grid.n_cells, dtype=np.uint8)
    if nphases > 1:
        fcum = 0.
        for iphase, phase in enumerate(phases):
            f = fvol[phase]
            grange = [fcum*ngrains, (fcum+f)*ngrains]
            fcum += f

            whr = (grains>=grange[0])*(grains<grange[1])
            grid.cell_data['phase'][whr] = phase

    grid.qarray = np.zeros((grid.n_cells, 4), dtype=DTYPEf)

    mydtype = [('grain', 'i4'), ('phase', 'i4'), ('npix', 'i4'), ('fvol', 'f4'),
               ('phi1', 'f4'), ('Phi', 'f4'), ('phi2', 'f4')]

    unic, indices, rindices, counts = np.unique(grains, return_index=True, return_inverse=True, return_counts=True)

    #grdata = np.rec.array(np.zeros(ngrains, dtype=mydtype))
    grdata = np.rec.array(np.zeros(len(unic), dtype=mydtype))
    phasedata = np.rec.array(np.zeros(nphases, dtype=mydtype[1:4]))
    if len(unic) < ngrains:
        logging.warning("The number of unique grain {} is smaller than the number of initial seeds {}.".format(len(unic), ngrains))
        ngrains = len(unic)
        newu = np.arange(ngrains)
        grains = newu[rindices]
        unic = newu

    qseeds = q4np.q4_random(ngrains)
    eulseeds = q4np.q4_to_eul(qseeds)
    if theta_spread is None:
        grid.qarray = qseeds[grains,:]
    else:
        if spread_type=='radial':
            xyzG = np.zeros((len(unic),3), dtype=DTYPEf)
            xyzG[:,0] = ndi.mean(xyzCells[:,0], labels=grains, index=unic)
            xyzG[:,1] = ndi.mean(xyzCells[:,1], labels=grains, index=unic)
            xyzG[:,2] = ndi.mean(xyzCells[:,2], labels=grains, index=unic)

            #vec = xyzCells - seeds[grains,:]
            vec = xyzCells - xyzG[grains,:]
            dist2seed = np.sqrt(np.sum(vec**2, axis=1))
            vec /= dist2seed[..., np.newaxis]
            qspread = np.zeros_like(grid.qarray)

            #if grid.params.dim3D:
            #    grSize = (counts*spacing**3)**(1./3)
            #else:
            #    grSize = (counts*spacing**2)**(1./2)
            #theta = np.radians(dist2seed / (grSize[grains]/2) * theta_spread)
            mxDist = ndi.maximum(dist2seed, labels=grains, index=unic)
            theta = np.radians(dist2seed / mxDist[grains] * theta_spread)
            grid.cell_data['theta_spread'] = np.degrees(theta)

            qspread[:,0] = np.cos(theta/2)
            qspread[:,1] = vec[:,0]*np.sin(theta/2)
            qspread[:,2] = vec[:,1]*np.sin(theta/2)
            qspread[:,3] = vec[:,2]*np.sin(theta/2)
            norm = np.sqrt(np.sum(qspread**2, axis=1))
            qspread /= norm[..., np.newaxis]
            grid.qarray = q4np.q4_mult(qspread, qseeds[grains,:]) # both expressed in the ref. frame
            #grid.qarray = q4np.q4_mult(qseeds[grains,:], qspread) # qsread expressed in the cryst. frame
        else:
            # random spread:
            qspread = q4np.q4_orispread(ncrys=len(grains), thetamax=theta_spread, misori=True)
            grid.qarray = q4np.q4_mult(qseeds[grains,:], qspread)

    grdata.grain = unic + 1
    grdata.phase = grid.cell_data['phase'][indices]
    grdata.npix = counts
    grdata.fvol = counts.astype(np.float32)/grid.n_cells
    grdata.phi1 = eulseeds[:,0]
    grdata.Phi  = eulseeds[:,1]
    grdata.phi2 = eulseeds[:,2]

    for iphase, phase in enumerate(phases):
        phasedata.phase[iphase] = phase
        phasedata.npix[iphase] = np.sum(grdata.npix[grdata.phase == phase])
        phasedata.fvol[iphase] = np.sum(grdata.fvol[grdata.phase == phase])

    logging.info("Actual volume fraction: {}".format(phasedata))

    np.savetxt('voro-{}-{}-{}-grains.txt'.format(dimX,dimY,dimZ), grdata,
               fmt="%6i %4i %9i %10.6f %10.3f %10.3f %10.3f", header=str(grdata.dtype.names))

    grid.cell_data['eul'] = q4np.q4_to_eul(grid.qarray)

    grid.save(grid.params.filename)
    grid._save_phase_info()

    logging.info("#### Finished to build virtual microstructure. ####")

    return grid

if __name__ == "__main__":
    import doctest
    doctest.testmod()
