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
from scipy.spatial import KDTree

import quaternions_np as q4np
import orientation_map as om

DTYPEf = np.float32
DTYPEi = np.int32

def Voronoi_microstructure(dimensions=(128,128,128), spacing=1,
                           ngrains=5**3, phases=[1,2], fvol=None):
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
    fvol : float, default=None
        dictionary of volume fractions for each phase label (if multiphase).

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
    >>> mic = Voronoi_microstructure(dimensions=(128,128,128), spacing=1, ngrains=5**3, phases=[0,1,2,4], fvol={0:0.15, 1:0.4, 2:0.4, 4:0.05})
    >>> mic = Voronoi_microstructure(dimensions=(128,128,128), spacing=1, ngrains=5**3, phases=[1,5])
    """

    dimX = dimensions[0]; dimY = dimensions[1]; dimZ = dimensions[2]
    phases = np.atleast_1d(phases).astype(np.uint8)
    phases.sort()
    nphases = len(phases)

    # check consistent phase input:
    if nphases > 1:
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

    # pyvista Image object:
    grid = om.OriMap(dimensions=(dimX+1, dimY+1, dimZ+1),
                     spacing=(spacing, spacing, spacing),
                     phase_to_crys )

    # coordinates of cell centers (starting at zero):
    whr = (grid.points[:,0] < grid.bounds[1])*\
          (grid.points[:,1] < grid.bounds[3])*\
          (grid.points[:,2] < grid.bounds[5])
    xyzCells = grid.points[whr]
    if not grid.n_cells == len(xyzCells):
        logging.error("n_cells does not match with the number of calculated cell centers.")

    seeds = np.random.rand(ngrains,3)
    seeds[:,0] *= dimX
    seeds[:,1] *= dimY
    seeds[:,2] *= dimZ

    tree = KDTree(seeds)
    dist, grains = tree.query(xyzCells)
    grid.cell_data['grain'] = grains + 1

    grid.cell_data['phase'] = np.ones(grid.n_cells, dtype=np.uint8)
    if nphases > 1:
        fcum = 0.
        for iphase, phase in enumerate(phases):
            f = fvol[phase]
            grange = [fcum*ngrains, (fcum+f)*ngrains]
            fcum += f

            whr = (grains>=grange[0])*(grains<grange[1])
            grid.cell_data['phase'][whr] = phase

    qseeds = q4np.q4_random(ngrains)
    eulseeds = q4np.q4_to_eul(qseeds)

    grid.qarray = np.zeros((grid.n_cells, 4), dtype=DTYPEf)

    mydtype = [('grain', 'i4'), ('phase', 'i4'), ('npix', 'i4'), ('fvol', 'f4'),
               ('phi1', 'f4'), ('Phi', 'f4'), ('phi2', 'f4')]
    grdata = np.rec.array(np.zeros(ngrains, dtype=mydtype))
    phasedata = np.rec.array(np.zeros(nphases, dtype=mydtype[1:4]))
    for igr in np.unique(grains):
        whrgr = (grains == igr)
        grid.qarray[whrgr] = qseeds[igr]

        grdata.grain[igr] = igr + 1
        grdata.phase[igr] = grid.cell_data['phase'][whrgr][0]
        nnn = np.sum(whrgr)
        grdata.npix[igr] = nnn
        grdata.fvol[igr] = float(nnn)/grid.n_cells
        grdata.phi1[igr] = eulseeds[igr,0]
        grdata.Phi[igr]  = eulseeds[igr,1]
        grdata.phi2[igr] = eulseeds[igr,2]

    for iphase, phase in enumerate(phases):
        phasedata.phase[iphase] = phase
        phasedata.npix[iphase] = np.sum(grdata.npix[grdata.phase == phase])
        phasedata.fvol[iphase] = np.sum(grdata.fvol[grdata.phase == phase])

    logging.info("Actual volume fraction: {}".format(phasedata))

    np.savetxt('voro-{}-{}-{}-grains.txt'.format(dimX,dimY,dimZ), grdata,
               fmt="%6i %4i %9i %10.6f %10.3f %10.3f %10.3f", header=str(grdata.dtype.names))

    grid.cell_data['eul'] = q4np.q4_to_eul(grid.qarray)

    grid.save('voro-{}-{}-{}.vtk'.format(dimX,dimY,dimZ))

    return grid

if __name__ == "__main__":
    import doctest
    doctest.testmod()
