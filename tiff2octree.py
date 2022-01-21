
import argparse
import fnmatch
import glob
import json
import logging
import os
import random
import sys
import time
import uuid
import warnings
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional, Union

import dask
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import dask_image.imread
from dask.distributed import Client, progress
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler, visualize
import numpy as np
import pandas as pd
from skimage import io, util
from skimage.transform import resize, downscale_local_mean
from dask_jobqueue import LSFCluster
from distributed import LocalCluster
from tifffile import TiffFile
from multiprocessing.pool import ThreadPool

from shutil import which
from typing import Union, List, Dict, Any, Optional
import warnings

dask.config.set({"jobqueue.lsf.use-stdin": True})

threading_env_vars = [
    "NUM_MKL_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENMP_NUM_THREADS",
    "OMP_NUM_THREADS",
]

def make_single_threaded_env_vars(threads: int) -> List[str]:
    return [f"export {var}={threads}" for var in threading_env_vars]


def bsub_available() -> bool:
    """Check if the `bsub` shell command is available
    Returns True if the `bsub` command is available on the path, False otherwise. This is used to check whether code is
    running on the Janelia Compute Cluster.
    """
    result = which("bsub") is not None
    return result


def get_LSFCLuster(
    threads_per_worker: int = 1,
    walltime: str = "1:00",
    death_timeout: str = "600s",
    **kwargs,
) -> LSFCluster:
    """Create a dask_jobqueue.LSFCluster for use on the Janelia Research Campus compute cluster.
    This function wraps the class dask_jobqueue.LSFCLuster and instantiates this class with some sensible defaults,
    given how the Janelia cluster is configured.
    This function will add environment variables that prevent libraries (OPENMP, MKL, BLAS) from running multithreaded routines with parallelism
    that exceeds the number of requested cores.
    Additional keyword arguments added to this function will be passed to the dask_jobqueue.LSFCluster constructor.
    Parameters
    ----------
    threads_per_worker: int
        Number of cores to request from LSF. Directly translated to the `cores` kwarg to LSFCluster.
    walltime: str
        The expected lifetime of a worker. Defaults to one hour, i.e. "1:00"
    cores: int
        The number of cores to request per worker. Defaults to 1.
    death_timeout: str
        The duration for the scheduler to wait for workers before flagging them as dead, e.g. "600s". For jobs with a large number of workers,
        LSF may take a long time (minutes) to request workers. This timeout value must exceed that duration, otherwise the scheduler will
        flag these slow-to-arrive workers as unresponsive and kill them.
    **kwargs:
        Additional keyword arguments passed to the LSFCluster constructor
    Examples
    --------
    >>> cluster = get_LSFCLuster(cores=2, project="scicompsoft", queue="normal")
    """

    if "env_extra" not in kwargs:
        kwargs["env_extra"] = []

    kwargs["env_extra"].extend(make_single_threaded_env_vars(threads_per_worker))

    USER = os.environ["USER"]
    HOME = os.environ["HOME"]

    if "local_directory" not in kwargs:
        # The default local scratch directory on the Janelia Cluster
        kwargs["local_directory"] = f"/scratch/{USER}/"

    if "log_directory" not in kwargs:
        log_dir = f"{HOME}/.dask_distributed/"
        Path(log_dir).mkdir(parents=False, exist_ok=True)
        kwargs["log_directory"] = log_dir

    cluster = LSFCluster(
        cores=threads_per_worker,
        walltime=walltime,
        death_timeout=death_timeout,
        **kwargs,
    )
    return cluster


def get_LocalCluster(threads_per_worker: int = 1, n_workers: int = 0, **kwargs):
    """
    Creata a distributed.LocalCluster with defaults that make it more similar to a deployment on the Janelia Compute cluster.
    This function is a light wrapper around the distributed.LocalCluster constructor.
    Parameters
    ----------
    n_workers: int
        The number of workers to start the cluster with. This defaults to 0 here.
    threads_per_worker: int
        The number of threads to assign to each worker.
    **kwargs:
        Additional keyword arguments passed to the LocalCluster constructor
    Examples
    --------
    >>> cluster = get_LocalCluster(threads_per_worker=8)
    """
    return LocalCluster(
        n_workers=n_workers, threads_per_worker=threads_per_worker, **kwargs
    )


def get_cluster(
    threads_per_worker: int = 1,
    deployment: Optional[str] = None,
    local_kwargs: Dict[str, Any] = {},
    lsf_kwargs: Dict[str, Any] = {"memory": "16GB"},
) -> Union[LSFCluster, LocalCluster]:

    """Convenience function to generate a dask cluster on either a local machine or the compute cluster.
    Create a distributed.Client object backed by either a dask_jobqueue.LSFCluster (for use on the Janelia Compute Cluster)
    or a distributed.LocalCluster (for use on a single machine). This function uses the output of the bsubAvailable function
    to determine whether code is running on the compute cluster or not.
    Additional keyword arguments given to this function will be forwarded to the constructor for the Client object.
    Parameters
    ----------
    threads_per_worker: int
        Number of threads per worker. Defaults to 1.
    deployment: str or None
        Which deployment (LocalCluster or LSFCluster) to prefer. If deployment=None, then LSFCluster is preferred, but LocalCluster is used if
        bsub is not available. If deployment='lsf' and bsub is not available, an error is raised.
    local_kwargs: dict
        Dictionary of keyword arguments for the distributed.LocalCluster constructor
    lsf_kwargs: dict
        Dictionary of keyword arguments for the dask_jobqueue.LSFCluster constructor
    """

    if "cores" in lsf_kwargs:
        warnings.warn(
            "The `cores` kwarg for LSFCLuster has no effect. Use the `threads_per_worker` argument instead."
        )

    if "threads_per_worker" in local_kwargs:
        warnings.warn(
            "the `threads_per_worker` kwarg was found in `local_kwargs`. It will be overwritten with the `threads_per_worker` argument to this function."
        )

    if deployment is None:
        if bsub_available():
            cluster = get_LSFCLuster(threads_per_worker, **lsf_kwargs)
        else:
            cluster = get_LocalCluster(threads_per_worker, **local_kwargs)
    elif deployment == "lsf":
        if bsub_available():
            cluster = get_LSFCLuster(threads_per_worker, **lsf_kwargs)
        else:
            raise EnvironmentError(
                "You requested an LSFCluster but the command `bsub` is not available."
            )
    elif deployment == "local":
        cluster = get_LocalCluster(threads_per_worker, **local_kwargs)
    else:
        raise ValueError(
            f'deployment must be one of (None, "lsf", or "local"), not {deployment}'
        )

    return cluster

def get_crop_from_tiled_tif(page, i0, j0, h, w):
    """Extract a crop from a TIFF image file directory (IFD).
    
    Only the tiles englobing the crop area are loaded and not the whole page.
    This is usefull for large Whole slide images that can't fit int RAM.
    Parameters
    ----------
    page : TiffPage
        TIFF image file directory (IFD) from which the crop must be extracted.
    i0, j0: int
        Coordinates of the top left corner of the desired crop.
    h: int
        Desired crop height.
    w: int
        Desired crop width.
    Returns
    -------
    out : ndarray of shape (imagedepth, h, w, sampleperpixel)
        Extracted crop.
    """

    if not page.is_tiled:
        raise ValueError("Input page must be tiled.")

    im_width = page.imagewidth
    im_height = page.imagelength

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    if i0 < 0 or j0 < 0 or i0 + h >= im_height or j0 + w >= im_width:
        raise ValueError("Requested crop area is out of image bounds.")

    tile_width, tile_height = page.tilewidth, page.tilelength
    i1, j1 = i0 + h, j0 + w

    tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
    tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

    tile_per_line = int(np.ceil(im_width / tile_width))

    out = np.empty((page.imagedepth,
                    (tile_i1 - tile_i0) * tile_height,
                    (tile_j1 - tile_j0) * tile_width,
                    page.samplesperpixel), dtype=page.dtype)

    fh = page.parent.filehandle

    jpegtables = page.tags.get('JPEGTables', None)
    if jpegtables is not None:
        jpegtables = jpegtables.value

    for i in range(tile_i0, tile_i1):
        for j in range(tile_j0, tile_j1):
            index = int(i * tile_per_line + j)

            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            fh.seek(offset)
            data = fh.read(bytecount)
            tile, indices, shape = page.decode(data, index, jpegtables)

            im_i = (i - tile_i0) * tile_height
            im_j = (j - tile_j0) * tile_width
            out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

    im_i0 = i0 - tile_i0 * tile_height
    im_j0 = j0 - tile_j0 * tile_width

    return out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :]

def get_untiled_crop(page, i0, j0, h, w):
    if page.is_tiled:
        raise ValueError("Input page must not be tiled")
    
    im_width = page.imagewidth
    im_height = page.imagelength
    
    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    i1, j1 = i0 + h, j0 + w
    if i0 < 0 or j0 < 0 or i1 >= im_height or j1 >= im_width:
        raise ValueError(f"Requested crop area is out of image bounds.{i0}_{i1}_{im_height}, {j0}_{j1}_{im_width}")
        
    out = np.empty((page.imagedepth, h, w, page.samplesperpixel), dtype=np.uint8)
    fh = page.parent.filehandle
    
    for index in range(i0, i1):
        offset = page.dataoffsets[index]
        bytecount = page.databytecounts[index]

        fh.seek(offset)
        data = fh.read(bytecount)

        tile, indices, shape = page.decode(data, index)
        
        out[:,index-i0,:,:] = tile[:,:,j0:j1,:]
    
    return out

def get_crop_tif_stack(fpath ,s0, i0, j0, d, h, w):
    tf = TiffFile(fpath)

    out = []

    for page in tf.pages:
        if page.is_tiled:
            out.append(get_crop_from_tiled_tif(page, i0, j0, h, w))
        else:
            out.append(get_untiled_crop(page, i0, j0, h, w))
    return out

# 2nd brightest of the 8 pixels
# equivalent to sort(vec(arg))[7] but half the time and a third the memory usage
def downsampling_function(view):
    m0 = 0
    m1 = 0
    for z in range(0, 2):
        for y in range(0, 2):
            for x in range(0, 2):
                tmp = view[z, y, x]
                if tmp > m0:
                    m1 = m0
                    m0 = tmp
                elif tmp > m1:
                    m1 = tmp
    return m1

def downsample(out_tile_jl, coord, shape_leaf_px, scratch):
    iy = ((coord - 1) >> 1) & 1 * shape_leaf_px[1] >> 1
    ix = ((coord - 1) >> 0) & 1 * shape_leaf_px[2] >> 1
    iz = ((coord - 1) >> 2) & 1 * shape_leaf_px[3] >> 1
    for z in range(1, shape_leaf_px[3] - 1, 2):
        tmpz = iz + (z + 1) >> 1
        for x in range(1, shape_leaf_px[2] - 1, 2):
            tmpx = ix + (x + 1) >> 1
            for y in range(1, shape_leaf_px[1] - 1, 2):
                tmpy = iy + (y + 1) >> 1
                out_tile_jl[tmpy, tmpx, tmpz] = downsampling_function(scratch[y:y + 1, x:x + 1, z:z + 1])

def save_data(path, image):
    io.imsave(path, image)

def octree_division(jobs, ch, nlevels, targetpath, dim_leaf, relpath, img_view):
        morton = relpath.split(os.path.sep)
        level = 1 if relpath == "" else (morton.length + 1)
        print("Level: "+str(level)+", Current path: "+str(targetpath)+"/"+str(relpath))
        Path(os.path.join(targetpath, relpath)).mkdir(parents=True, exist_ok=True)
        if level < nlevels:
            img_down_next = np.zeros((3, 2))
            dim_view = np.array([img_view[0][1] - img_view[0][0], img_view[1][1] - img_view[1][0], img_view[2][1] - img_view[2][0]])
            for z in range(1,3):
                for y in range(1,3):
                    for x in range(1,3):
                        octant_path = str((x + 2 * (y - 1) + 4 * (z - 1)))
                        start_x = 1 if x == 1 else dim_view[2] >> 1 + 1
                        end_x = dim_view[2] >> 1 if x == 1 else dim_view[2]
                        start_y = 1 if y == 1 else dim_view[1] >> 1 + 1
                        end_y = dim_view[1] >> 1 if y == 1 else dim_view[1]
                        start_z = 1 if z == 1 else dim_view[3] >> 1 + 1
                        end_z = dim_view[3] >> 1 if z == 1 else dim_view[3]
      
            print("octant:({0},{1},{2}) -> {3}/{4} ({5}:{6}, {7}:{8}, {9}:{10})".format(x, y, z, relpath, octant_path, start_x, end_x, start_y, end_y, start_z, end_z))
            new_view = np.array([start_x, end_x], [start_y, end_y], [start_z, end_z])
            octree_division(jobs, ch, targetpath, os.path.join(relpath, octant_path), new_view)
        
        saveme = img_view if level == nlevels else img_down_next
        filename = "default.{0}.tif".format(ch)
        save_data(os.path.join(targetpath, relpath, filename), saveme)
        if (level > 1):
            downsample(os.path.join(targetpath, relpath, filename), dim_leaf, saveme)


def downsample_2ndmax(out_tile_jl, coord, shape_leaf_px, scratch):
    ix = (((coord - 1) >> 0) & 1) * shape_leaf_px[2] >> 1
    iy = (((coord - 1) >> 1) & 1) * shape_leaf_px[1] >> 1
    iz = (((coord - 1) >> 2) & 1) * shape_leaf_px[0] >> 1
    for z in range(0, shape_leaf_px[0]-1, 2):
        tmpz = iz + ((z + 1) >> 1)
        for y in range(0, shape_leaf_px[1]-1, 2):
            tmpy = iy + ((y + 1) >> 1)
            for x in range(0, shape_leaf_px[2]-1, 2):
                tmpx = ix + ((x + 1) >> 1)
                out_tile_jl[tmpz, tmpy, tmpx] = downsampling_function(scratch[z:z + 2, y:y + 2, x:x + 2])

def downsample_aa(out_tile_jl, coord, shape_leaf_px, scratch):
    ix = (((coord - 1) >> 0) & 1) * shape_leaf_px[2] >> 1
    iy = (((coord - 1) >> 1) & 1) * shape_leaf_px[1] >> 1
    iz = (((coord - 1) >> 2) & 1) * shape_leaf_px[0] >> 1
    ixx = ix+(shape_leaf_px[2] >> 1)
    iyy = iy+(shape_leaf_px[1] >> 1)
    izz = iz+(shape_leaf_px[0] >> 1)
    if scratch.dtype is np.dtype('uint8'):
        out_tile_jl[iz:izz, iy:iyy, ix:ixx] = util.img_as_ubyte(resize(scratch, (shape_leaf_px[0]>>1, shape_leaf_px[1]>>1, shape_leaf_px[2]>>1), anti_aliasing=True))
    elif scratch.dtype is np.dtype('uint16'):
        out_tile_jl[iz:izz, iy:iyy, ix:ixx] = util.img_as_uint(resize(scratch, (shape_leaf_px[0]>>1, shape_leaf_px[1]>>1, shape_leaf_px[2]>>1), anti_aliasing=True))
    elif scratch.dtype is np.dtype('float32'):
        out_tile_jl[iz:izz, iy:iyy, ix:ixx] = util.img_as_float32(resize(scratch, (shape_leaf_px[0]>>1, shape_leaf_px[1]>>1, shape_leaf_px[2]>>1), anti_aliasing=True))

def downsample_area(out_tile_jl, coord, shape_leaf_px, scratch):
    ix = (((coord - 1) >> 0) & 1) * shape_leaf_px[2] >> 1
    iy = (((coord - 1) >> 1) & 1) * shape_leaf_px[1] >> 1
    iz = (((coord - 1) >> 2) & 1) * shape_leaf_px[0] >> 1
    ixx = ix+(shape_leaf_px[2] >> 1)
    iyy = iy+(shape_leaf_px[1] >> 1)
    izz = iz+(shape_leaf_px[0] >> 1)
    down_img = downscale_local_mean(scratch, (2, 2 ,2))
    out_tile_jl[iz:izz, iy:iyy, ix:ixx] = down_img.astype(scratch.dtype)

def get_octree_relative_path(chunk_coord, nlevels, level):
    relpath = ''
    pos = np.asarray(chunk_coord)
    for lv in range(nlevels-1, nlevels-level-1, -1):
        d = pow(2, lv)
        octant_path = str(1 + int(pos[2] / d) + 2 * int(pos[1] / d) + 4 * int(pos[0] / d))
        relpath = os.path.join(relpath, octant_path)
        pos[2] = pos[2] - int(pos[2] / d) * d
        pos[1] = pos[1] - int(pos[1] / d) * d
        pos[0] = pos[0] - int(pos[0] / d) * d
    
    return relpath

def save_block(chunk, target_path, nlevels, dim_leaf, ch, block_id=None):
    if block_id == None:
        return np.array(0)[None, None, None]
    
    relpath = get_octree_relative_path(block_id, nlevels, nlevels)

    dir_path = os.path.join(target_path, relpath)
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    full_path = os.path.join(dir_path, "default.{0}.tif".format(ch))

    print(full_path)

    io.imsave(full_path, chunk)

    return np.array(block_id[0])[None, None, None] if block_id != None else np.array(0)[None, None, None]

def downsample_and_save_block(chunk_coord, target_path, nlevels, level, dim_leaf, ch, type, downsampling_method):

    relpath = get_octree_relative_path(chunk_coord, nlevels, level)

    dir_path = os.path.join(target_path, relpath)
    img_name = "default.{0}.tif".format(ch)
    
    img_down = np.zeros(dim_leaf, dtype=type)
    
    for oct in range(1, 9):
        blk_path = os.path.join(dir_path, str(oct))
        scratch = io.imread(os.path.join(blk_path, img_name))
        if downsampling_method == 'area':
            downsample_area(img_down, oct, dim_leaf, scratch)
        elif downsampling_method == 'aa':
            downsample_aa(img_down, oct, dim_leaf, scratch)
        else:
            downsample_2ndmax(img_down, oct, dim_leaf, scratch)

    full_path = os.path.join(dir_path, "default.{0}.tif".format(ch))

    print(full_path)

    io.imsave(full_path, img_down)


def build_octree_from_tiff_slices():
    argv = sys.argv
    argv = argv[1:]

    usage_text = ("Usage:" + "  slice2octree.py" + " [options]")
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("-t", "--thread", dest="number", type=int, default=16, help="number of threads")
    parser.add_argument("-i", "--inputdir", dest="input", type=str, default=None, help="input directory")
    parser.add_argument("-f", "--inputfile", dest="file", type=str, default=None, help="input image stack")
    parser.add_argument("-o", "--output", dest="output", type=str, default="", help="output directory")
    parser.add_argument("-l", "--level", dest="level", type=int, default=1, help="number of levels")
    parser.add_argument("-c", "--channel", dest="channel", type=int, default=0, help="channel id")
    parser.add_argument("-d", "--downsample", dest="downsample", type=str, default='area', help="downsample method: 2ndmax, area, aa (anti-aliasing)")
    parser.add_argument("-m", "--monitor", dest="monitor", default=False, action="store_true", help="activate monitoring")
    parser.add_argument("--memory", dest="memory", type=str, default=None, help="memory amount per thread (for LSF cluster)")
    parser.add_argument("--project", dest="project", type=str, default=None, help="project name (for LSF cluster)")
    parser.add_argument("--maxjobs", dest="maxjobs", type=int, default=16, help="maximum jobs (for LSF cluster)")
    parser.add_argument("--lsf", dest="lsf", default=False, action="store_true", help="use LSF cluster")

    if not argv:
        parser.print_help()
        exit()

    args = parser.parse_args(argv)

    tnum = args.number
    indir = args.input
    infile = args.file
    outdir = args.output
    nlevels = args.level
    ch = args.channel
    dmethod = args.downsample
    monitoring = args.monitor

    my_lsf_kwargs={}
    if args.memory:
       my_lsf_kwargs['mem'] = args.memory
    if args.project:
       my_lsf_kwargs['project'] = args.project
    
    cluster = None
    if args.lsf:
        cluster = get_cluster(deployment="lsf", lsf_kwargs = my_lsf_kwargs)
        cluster.scale(maximum_jobs = args.maxjobs)
    else:
        cluster = get_cluster(deployment="local")
    
    cluster.scale(tnum)

    dashboard_address = None
    if monitoring: 
        dashboard_address = ":8787"
        print(f"Starting dashboard on {dashboard_address}")
        client = Client(address=cluster, processes=True, dashboard_address=dashboard_address)
    else:
        client = Client(address=cluster)

    images = None
    if indir:
        images = dask_image.imread.imread(indir+'/*.tif')
    elif infile:
        images = dask_image.imread.imread(infile)
    else:
        print('Please specify an input dataset.')
        return
    dim = np.asarray(images.shape)
    print("Will generate octree with " + str(nlevels) +" levels to " + str(outdir))
    print("Image dimensions: " + str(dim))
    
    while(dim[0] % pow(2, nlevels) > 0):
        dim[0] -= 1
    while(dim[1] % pow(2, nlevels) > 0):
        dim[1] -= 1
    while(dim[2] % pow(2, nlevels) > 0):
        dim[2] -= 1
    dim_leaf = [x >> (nlevels - 1) for x in dim]

    print("Adjusted image size: " + str(dim) + ", Dim leaf: " + str(dim_leaf))
    adjusted = images[:dim[0], :dim[1], :dim[2]]
    volume = adjusted.rechunk(dim_leaf)

    ranges = [(c, c+dim_leaf[0]) for c in range(0,dim[0],dim_leaf[0])]

    #initial
    volume.map_blocks(save_block, outdir, nlevels, dim_leaf, ch, chunks=(1,1,1)).compute()

    #downsample
    for lv in range(nlevels-1, 0, -1):
        futures = []
        bnum = pow(2, lv - 1)
        for z in range(1, bnum+1):
            for y in range(1, bnum+1):
                for x in range(1, bnum+1):
                    future = dask.delayed(downsample_and_save_block)((z,y,x), outdir, nlevels, lv, dim_leaf, ch, volume.dtype, dmethod)
                    futures.append(future)
        with ProgressBar():
            dask.compute(futures)

    client.close()


def main():
    build_octree_from_tiff_slices()


if __name__ == '__main__':
    main()
