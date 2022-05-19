
import argparse
import fnmatch
import glob
import io
import re
import json
import logging
import os
import random
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk

import dask
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import dask_image.imread
from dask.distributed import Client, Variable
from distributed.diagnostics.progressbar import progress
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler, visualize
import numpy as np
import pandas as pd
import skimage
from skimage import util
from skimage.transform import resize, downscale_local_mean
from dask_jobqueue import LSFCluster
from distributed import LocalCluster
from tifffile import TiffFile
from multiprocessing.pool import ThreadPool

from shutil import which, copyfile, rmtree
from typing import Union, List, Dict, Any, Optional
import warnings

import ktx
from ktx.util import create_mipmaps, mipmap_dimension, interleave_channel_arrays, downsample_array_xy
from ktx.octree.ktx_from_rendered_tiff import RenderedMouseLightOctree, RenderedTiffBlock

import rasterio
from rasterio.windows import Window

from scipy import ndimage


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


def get_LocalCluster(threads_per_worker: int = 1, n_workers: int = 0, memory_limit: str = '16GB', **kwargs):
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
        n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit, **kwargs
    )


def get_cluster(
    threads_per_worker: int = 1,
    walltime: str = "1:00",
    local_memory_limit: str = "16GB",
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
            cluster = get_LSFCLuster(threads_per_worker, walltime, **lsf_kwargs)
        else:
            raise EnvironmentError(
                "You requested an LSFCluster but the command `bsub` is not available."
            )
    elif deployment == "local":
        cluster = get_LocalCluster(threads_per_worker, 0, local_memory_limit, **local_kwargs)
    else:
        raise ValueError(
            f'deployment must be one of (None, "lsf", or "local"), not {deployment}'
        )

    return cluster

# https://stackoverflow.com/questions/62567983/block-reduce-downsample-3d-array-with-mode-function
def blockify(image, block_size):
    shp = image.shape
    out_shp = [s//b for s,b in zip(shp, block_size)]
    reshape_shp = np.c_[out_shp,block_size].ravel()
    nC = np.prod(block_size)
    return image.reshape(reshape_shp).transpose(0,2,4,1,3,5).reshape(-1,nC)

def get_output_bbox_for_downsampling(coord, shape_leaf_px):
    ix = (((coord - 1) >> 0) & 1) * shape_leaf_px[2] >> 1
    iy = (((coord - 1) >> 1) & 1) * shape_leaf_px[1] >> 1
    iz = (((coord - 1) >> 2) & 1) * shape_leaf_px[0] >> 1
    ixx = ix+(shape_leaf_px[2] >> 1)
    iyy = iy+(shape_leaf_px[1] >> 1)
    izz = iz+(shape_leaf_px[0] >> 1)

    return np.array([[iz, iy, ix], [izz, iyy, ixx]])

def downsample_2ndmax(out_tile_jl, coord, shape_leaf_px, scratch):
    bbox = get_output_bbox_for_downsampling(coord, shape_leaf_px)
    b = blockify(scratch, block_size=(2,2,2))
    b.sort(axis=1)
    down_img = b[:,-2].reshape(bbox[1,0]-bbox[0,0], bbox[1,1]-bbox[0,1], bbox[1,2]-bbox[0,2])
    out_tile_jl[bbox[0,0]:bbox[1,0], bbox[0,1]:bbox[1,1], bbox[0,2]:bbox[1,2]] = down_img.astype(scratch.dtype)

def downsample_aa(out_tile_jl, coord, shape_leaf_px, scratch):
    bbox = get_output_bbox_for_downsampling(coord, shape_leaf_px)
    if scratch.dtype is np.dtype('uint8'):
        out_tile_jl[bbox[0,0]:bbox[1,0], bbox[0,1]:bbox[1,1], bbox[0,2]:bbox[1,2]] = util.img_as_ubyte(resize(scratch, (shape_leaf_px[0]>>1, shape_leaf_px[1]>>1, shape_leaf_px[2]>>1), anti_aliasing=True))
    elif scratch.dtype is np.dtype('uint16'):
        out_tile_jl[bbox[0,0]:bbox[1,0], bbox[0,1]:bbox[1,1], bbox[0,2]:bbox[1,2]] = util.img_as_uint(resize(scratch, (shape_leaf_px[0]>>1, shape_leaf_px[1]>>1, shape_leaf_px[2]>>1), anti_aliasing=True))
    elif scratch.dtype is np.dtype('float32'):
        out_tile_jl[bbox[0,0]:bbox[1,0], bbox[0,1]:bbox[1,1], bbox[0,2]:bbox[1,2]] = util.img_as_float32(resize(scratch, (shape_leaf_px[0]>>1, shape_leaf_px[1]>>1, shape_leaf_px[2]>>1), anti_aliasing=True))

def downsample_area(out_tile_jl, coord, shape_leaf_px, scratch):
    bbox = get_output_bbox_for_downsampling(coord, shape_leaf_px)
    down_img = downscale_local_mean(scratch, (2, 2 ,2))
    out_tile_jl[bbox[0,0]:bbox[1,0], bbox[0,1]:bbox[1,1], bbox[0,2]:bbox[1,2]] = down_img.astype(scratch.dtype)

def downsample_spline3(out_tile_jl, coord, shape_leaf_px, scratch):
    bbox = get_output_bbox_for_downsampling(coord, shape_leaf_px)
    down_img = ndimage.zoom(scratch, 0.5)
    out_tile_jl[bbox[0,0]:bbox[1,0], bbox[0,1]:bbox[1,1], bbox[0,2]:bbox[1,2]] = down_img.astype(scratch.dtype)

def get_octree_relative_path(chunk_coord, level, sep=True):
    relpath = ''
    pos = np.asarray(chunk_coord)
    pos = np.add(pos, np.full(3, -1))
    for lv in range(level-1, -1, -1):
        d = pow(2, lv)
        octant_path = str(1 + int(pos[2] / d) + 2 * int(pos[1] / d) + 4 * int(pos[0] / d))
        if lv < level-1:
            if sep:
                relpath = os.path.join(relpath, octant_path)
            else:
                relpath += octant_path
        pos[2] = pos[2] - int(pos[2] / d) * d
        pos[1] = pos[1] - int(pos[1] / d) * d
        pos[0] = pos[0] - int(pos[0] / d) * d
    
    return relpath

def get_cropped_image_rasterio(file_paths, z0, y0, x0, d, h, w, type, ch):
    output = np.zeros((d, h, w), dtype=type)
    for i in range(z0, z0 + d):
        if i - z0 < len(file_paths):
            try:
                with rasterio.open(file_paths[i - z0]) as src:
                    data = src.read(ch+1, window=Window(x0, y0, w, h))
                    output[i - z0, :h, :w] = data[:, :]
            except BaseException as err:
                logging.error(err)
    return output

def gen_blocks_from_slices_batch(chunk_coords, file_paths, target_path, nlevels, dim_leaf, ch, type, resolutions, chnum, resume):
    for pos in chunk_coords:
        gen_block_from_slices(pos, file_paths, target_path, nlevels, dim_leaf, ch, type, resolutions, chnum, resume)

def gen_block_from_slices(chunk_coord, file_paths, target_path, nlevels, dim_leaf, chid, type, resolutions, chnum, resume):
    relpath = get_octree_relative_path(chunk_coord, nlevels)

    dir_path = os.path.join(target_path, relpath)
    for ch in range(chnum):
        full_path = os.path.join(dir_path, "default.{0}.tif".format(chid+ch))
        if resume and os.path.exists(full_path):
            logging.info("skipped (exists): " + full_path)
            continue

        #logging.info((dim_leaf[0]*chunk_coord[0], dim_leaf[1]*chunk_coord[1], dim_leaf[2]*chunk_coord[2]))

        img_data = get_cropped_image_rasterio(file_paths, dim_leaf[0]*(chunk_coord[0]-1), dim_leaf[1]*(chunk_coord[1]-1), dim_leaf[2]*(chunk_coord[2]-1), dim_leaf[0], dim_leaf[1], dim_leaf[2], type, ch)

        if img_data.max() > 0:
            logging.info(full_path)
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            skimage.io.imsave(full_path, img_data, compress=6)
        else:
            logging.info("skipped (empty): " + full_path)

#return True if the block exists
def check_block(chunk_coord, target_path, nlevels, chid, chnum):
    relpath = get_octree_relative_path(chunk_coord, nlevels)

    dir_path = os.path.join(target_path, relpath)
    block_exists = True
    for ch in range(chnum):
        full_path = os.path.join(dir_path, "default.{0}.tif".format(chid+ch))
        if not os.path.exists(full_path):
            block_exists = False
    return block_exists

def save_block(chunk, target_path, nlevels, dim_leaf, ch, block_id=None):
    if block_id == None:
        return np.array(0)[None, None, None]
    
    relpath = get_octree_relative_path(block_id, nlevels)

    dir_path = os.path.join(target_path, relpath)
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    full_path = os.path.join(dir_path, "default.{0}.tif".format(ch))

    logging.info(full_path)

    skimage.io.imsave(full_path, chunk, compress=6)

    return np.array(block_id[0])[None, None, None] if block_id != None else np.array(0)[None, None, None]

def downsample_and_save_block_batch(chunk_coords, target_path, level, dim_leaf, ch, type, downsampling_method, resolutions, ktxdir, delete_source, resume):
    for pos in chunk_coords:
        downsample_and_save_block(pos, target_path, level, dim_leaf, ch, type, downsampling_method, resolutions, ktxdir, delete_source, resume)

def downsample_and_save_block(chunk_coord, target_path, level, dim_leaf, ch, type, downsampling_method, resolutions, ktxdir, delete_source, resume):

    relpath = get_octree_relative_path(chunk_coord, level)

    dir_path = os.path.join(target_path, relpath)
    img_name = "default.{0}.tif".format(ch)
    full_path = os.path.join(dir_path, img_name)

    if resume:
        if os.path.exists(full_path):
            return
        if ktxdir:
            ktx_name = 'block_8_xy_'+get_octree_relative_path(chunk_coord, level, False)+".ktx"
            if os.path.exists(os.path.join(ktxdir, ktx_name)):
                return
    
    img_down = np.zeros(dim_leaf, dtype=type)
    
    for oct in range(1, 9):
        blk_path = os.path.join(os.path.join(dir_path, str(oct)), img_name)
        try:
            scratch = skimage.io.imread(blk_path)
        except:
            logging.info("empty: " + blk_path)
            continue
        if downsampling_method == 'area':
            downsample_area(img_down, oct, dim_leaf, scratch)
        elif downsampling_method == 'aa':
            downsample_aa(img_down, oct, dim_leaf, scratch)
        elif downsampling_method == 'spline':
            downsample_spline3(img_down, oct, dim_leaf, scratch)
        else:
            downsample_2ndmax(img_down, oct, dim_leaf, scratch)
        if ktxdir:
            convert_block_ktx_path(os.path.join(relpath, str(oct)), target_path, ktxdir, level+1, True, True, True, delete_source)

    Path(dir_path).mkdir(parents=True, exist_ok=True)
    if level > 1:
        if img_down.max() > 0:
            logging.info(full_path)
            skimage.io.imsave(full_path, img_down, compress=6)
    else:
        logging.info(full_path)
        skimage.io.imsave(full_path, img_down)
        convert_block_ktx_path(relpath, target_path, ktxdir, level, True, True, True, False)

def convert_block_ktx_batch(chunk_coords, source_path, target_path, level, downsample_intensity, downsample_xy, make_dir, delete_source):
    for pos in chunk_coords:
        convert_block_ktx(pos, source_path, target_path, level, downsample_intensity, downsample_xy, make_dir, delete_source)

def convert_block_ktx(chunk_coord, source_path, target_path, level, downsample_intensity, downsample_xy, make_dir, delete_source):
    relpath = get_octree_relative_path(chunk_coord, level)
    convert_block_ktx_path(relpath, source_path, target_path, level, downsample_intensity, downsample_xy, make_dir, delete_source)

def convert_block_ktx_path(relpath, source_path, target_path, level, downsample_intensity, downsample_xy, make_dir, delete_source):
    tardir_path = os.path.join(target_path, relpath)
    if make_dir:
        Path(tardir_path).mkdir(parents=True, exist_ok=True)

    srcdir_path = os.path.join(source_path, relpath)
    file_list = glob.glob(os.path.join(srcdir_path, 'default.[0-9].tif'))
    if len(file_list) == 0:
        return

    octree_path0 = relpath.split(os.path.sep)
    octree_path = []
    for level_str in octree_path0:
        if re.match(r'[1-8]', level_str):
            octree_path.append(int(level_str))
    
    o = RenderedMouseLightOctree(input_folder=source_path, downsample_intensity=downsample_intensity, downsample_xy=downsample_xy)
    block = RenderedTiffBlock(os.path.join(source_path, relpath), o, octree_path)
    
    file_name = 'block'
    if downsample_intensity:
        file_name += '_8'
    if downsample_xy:
        file_name += '_xy'
    opath1 = "".join([str(n) for n in block.octree_path])
    file_name += '_'+opath1+".ktx"
    output_dir = os.path.join(target_path, relpath)
    full_path = os.path.join(output_dir, file_name)
    
    logging.info(full_path)
    
    overwrite = True
    if os.path.exists(full_path) and not overwrite:
        file_size = os.stat(full_path).st_size
        if file_size > 0:
            logging.info("Skipping an existing file %s" % full_path)
            return

    try:
        f = open(full_path, 'wb')
        block.write_ktx_file(f)
        f.flush()
        f.close()
    except IOError as err:
        logging.error(err)
        logging.error("Error writing file %s" % full_path)
        f.flush()
        f.close()
        os.unlink(f.name)

    if delete_source and level > 1:
        try:
            for fpath in glob.glob(os.path.join(srcdir_path, 'default.[0-9].tif')):
                os.remove(fpath)
        except OSError as err:
            logging.error(err)
            logging.error("Could not delete %s" % fpath)

def conv_tiled_tiff(input, output, tilesize, resume):
    if not os.path.exists(input):
        return input

    if resume and os.path.exists(output):
        return output

    is_tiled = True
    try:    
        tif = TiffFile(input)
        is_tiled = tif.pages[0].is_tiled
        tif.close()
    except Exception:
        return input

    if not is_tiled:
        try:
            img = skimage.io.imread(input)
            skimage.io.imsave(output, img, compress=6, tile=tilesize)
            logging.info("saved tiled-tiff: " + output)
        except Exception:
            logging.error("failed to convert: " + output)
            return input
    
    return output

def conv_tiled_tiffs(input_list, outdir, tilesize, resume):
    ret_list = []
    for fpath in input_list:
        fname = os.path.basename(fpath)
        ret = conv_tiled_tiff(fpath, os.path.join(outdir, fname), tilesize, resume)
        ret_list.append(ret)
    return ret_list

def delete_files(target_list):
    for fpath in target_list:
        try:
            os.remove(fpath)
        except:
            logging.error("Error while deleting file ", fpath)

def scantree(path):
    for entry in scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry

def setup_cluster(
    cluster_address: str = None,
    monitoring: bool = False,
    is_lsf: bool = False,
    walltime: str = "1:00",
    memory_limit: str = "16GB",
    lsf_maximum_jobs: int = 1,
    thread_num: int = 1,
    project: str = ""
    ):

    cluster = None
    if cluster_address:
        cluster = cluster_address
    elif is_lsf:
        my_lsf_kwargs={}
        my_lsf_kwargs['memory'] = memory_limit
        if project:
            my_lsf_kwargs['project'] = project
        cluster = get_cluster(deployment="lsf", walltime=walltime, lsf_kwargs = my_lsf_kwargs)
        cluster.adapt(minimum_jobs=1, maximum_jobs = lsf_maximum_jobs)
        cluster.scale(thread_num)
    else:
        cluster = get_cluster(deployment="local", local_memory_limit = memory_limit)
        cluster.scale(thread_num)

    dashboard_address = None
    if monitoring: 
        dashboard_address = ":8787"
        print(f"Starting dashboard on {dashboard_address}")
        client = Client(address=cluster, processes=True, dashboard_address=dashboard_address)
    else:
        client = Client(address=cluster)

    return client

def adjust_dimensions(dim, nlevels):
    ret = np.copy(dim)
    for i in range(0, len(ret)):
        while(ret[i] % pow(2, nlevels) > 0):
            ret[i] -= 1
    return ret

def stack_to_dask_array(
    file_path: str,
    nlevels: int
    ):
    ret = None
    if file_path:
        img = dask_image.imread.imread(file_path)

        dim = np.asarray(img.shape)
        dim = adjust_dimensions(dim, nlevels)
        dim_leaf = [x >> (nlevels - 1) for x in dim]

        logging.info("Adjusted image size: " + str(dim) + ", Dim leaf: " + str(dim_leaf))

        adjusted = img[:dim[0], :dim[1], :dim[2]]
        ret = adjusted.rechunk(dim_leaf)
    return ret

def slice_to_dask_array(
    indir: str,
    nlevels: int
    ):
    ret = None
    images = None
    dim = None
    volume_dtype = None
    if indir:
        img_files = [os.path.join(indir, f) for f in os.listdir(indir) if f.endswith(('.tif', '.jp2'))]
        im_width = 0
        im_height = 0
        samples_per_pixel = 1
        if img_files[0].endswith('.tif'):
            with TiffFile(img_files[0]) as tif:
                im_width = tif.pages[0].imagewidth
                im_height = tif.pages[0].imagelength
                logging.info(tif.pages[0].dtype)
                volume_dtype = tif.pages[0].dtype
        elif img_files[0].endswith('.jp2'):
            with rasterio.open(img_files[0]) as src:
                im_width = src.width
                im_height = src.height
                samples_per_pixel = src.count
                volume_dtype = src.dtypes[0]

        dim = np.asarray((len(img_files), im_height, im_width))
        dim = adjust_dimensions(dim, nlevels)
        dim_leaf = [x >> (nlevels - 1) for x in dim]

        logging.info("Adjusted image size: " + str(dim) + ", Dim leaf: " + str(dim_leaf))

        ret = da.zeros((dim[0], dim[1], dim[2], samples_per_pixel), chunks=(dim_leaf[0], dim_leaf[1], dim_leaf[2], samples_per_pixel), dtype=volume_dtype)
    
    return ret

def parse_voxel_size(voxel_size_str: str):
    vsstr = voxel_size_str.split(",")
    vs = []
    if len(vsstr) > 0:
        vs.append(float(vsstr[2]))
    else:
        vs.append(1.0)
    if len(vsstr) > 1:
        vs.append(float(vsstr[1]))
    else:
        vs.append(1.0)
    if len(vsstr) > 2:
        vs.append(float(vsstr[0]))
    else:
        vs.append(1.0)
    return vs

def save_transform_txt(
    outdir: str,
    origin: str,
    voxsize: str,
    nlevels: int,
    ktxout: str
    ):

    ostr = origin.split(",")
    o = []
    if len(ostr) > 0:
        o.append(ostr[2])
    else:
        o.append("0.0")
    if len(ostr) > 1:
        o.append(ostr[1])
    else:
        o.append("0.0")
    if len(ostr) > 2:
        o.append(ostr[0])
    else:
        o.append("0.0")

    vsstr = voxsize.split(",")
    vs = []
    if len(vsstr) > 0:
        vs.append(float(vsstr[2]))
    else:
        vs.append(1.0)
    if len(vsstr) > 1:
        vs.append(float(vsstr[1]))
    else:
        vs.append(1.0)
    if len(vsstr) > 2:
        vs.append(float(vsstr[0]))
    else:
        vs.append(1.0)

    l = []
    l.append("ox: " + o[2])
    l.append("oy: " + o[1])
    l.append("oz: " + o[0])
    l.append("sx: " + '{:.14g}'.format(vs[2] * 1000 * pow(2, nlevels-1)))
    l.append("sy: " + '{:.14g}'.format(vs[1] * 1000 * pow(2, nlevels-1)))
    l.append("sz: " + '{:.14g}'.format(vs[0] * 1000 * pow(2, nlevels-1)))
    l.append("nl: " + str(nlevels))

    Path(outdir).mkdir(parents=True, exist_ok=True)
    tr_path = os.path.join(outdir, "transform.txt")
    with open(tr_path, mode='w') as f:
        f.write('\n'.join(l))
    
    if ktxout:
        ktxroot = str(Path(ktxout).parent)
        if outdir != ktxroot:
            Path(ktxroot).mkdir(parents=True, exist_ok=True)
            copyfile(tr_path, os.path.join(ktxroot, "transform.txt"))
        if outdir != ktxout:
            Path(ktxout).mkdir(parents=True, exist_ok=True)
            copyfile(tr_path, os.path.join(ktxout, "transform.txt"))

def covert_tiff_to_tiled_tiff(input_paths: List[str], task_num: int, outdir: str, maxbatch: int = 100, resume: bool = False):
    slice_files = input_paths
    futures = []
    logging.info("tiff-to-tiled-tiff conversion")
    batch_slice_num = len(slice_files) / task_num
    if maxbatch > 0 and batch_slice_num > maxbatch:
        batch_slice_num = maxbatch
    if batch_slice_num < 1: 
        batch_slice_num = 1
    chunked_tif_list = np.array_split(slice_files, (int)(len(slice_files) / batch_slice_num))
    logging.info("tiff-to-tiled-tiff conversion: list " + str(len(chunked_tif_list)))
    for tlist in chunked_tif_list:
        #logging.info(tlist)
        future = dask.delayed(conv_tiled_tiffs)(tlist, outdir, (256, 256), resume)
        futures.append(future)
    with ProgressBar():
        results = dask.compute(futures)
        slice_files = [item for sublist in results[0] for item in sublist]
    logging.info("done")
    return slice_files

def delete_temporary_files(tmpdir: str, task_num: int, maxbatch: int = 200):
    logging.info("deleting temporary files...")
    tlist = [entry.path for entry in scantree(tmpdir)]
    if len(tlist) == 0:
        return
    batch_file_num = len(tlist) / task_num
    if maxbatch > 0 and batch_file_num > maxbatch:
        batch_file_num = maxbatch
    if batch_file_num < 1:
        batch_file_num = 1
    chunked_file_list = np.array_split(tlist, (int)(len(tlist) / batch_file_num))
    futures = []
    for flist in chunked_file_list:
        future = dask.delayed(delete_files)(flist)
        futures.append(future)
    with ProgressBar():
        dask.compute(futures)
    logging.info("done")

# loop over the chunks in the dask array.
def save_tiff_blocks(input_slices: List[str], output_path: str, z: int, nlevels: int, task_num: int, maxbatch: int, ch: int, voxel_size_str: str, darray: da.Array, resume: bool):
    futures = []
    bnum = pow(2, nlevels - 1)
    volume_dtype = darray.dtype
    samples_per_pixel = darray.shape[3]
    dim_leaf = darray.chunksize[:3]
    vs = parse_voxel_size(voxel_size_str)

    batch_block_num = (int)(bnum * bnum / task_num)
    if maxbatch > 0 and batch_block_num > maxbatch:
        batch_block_num = maxbatch
    if batch_block_num < 1: 
        batch_block_num = 1
    coord_list = []
    for y in range(1, bnum+1):
        for x in range(1, bnum+1):
            coord_list.append((z,y,x))
            if len(coord_list) >= batch_block_num:
                future = dask.delayed(gen_blocks_from_slices_batch)(coord_list, input_slices, output_path, nlevels, dim_leaf, ch, volume_dtype, vs, samples_per_pixel, resume)
                futures.append(future)
                coord_list = []
    if len(coord_list) > 0:
        future = dask.delayed(gen_blocks_from_slices_batch)(coord_list, input_slices, output_path, nlevels, dim_leaf, ch, volume_dtype, vs, samples_per_pixel, resume)
        futures.append(future)
    with ProgressBar():
        dask.compute(futures)

def check_tiff_blocks(output_path: str, z: int, nlevels: int, ch: int, darray: da.Array):
    logging.info("resume: checking tiff blocks")
    futures = []
    bnum = pow(2, nlevels - 1)
    samples_per_pixel = darray.shape[3]
    for y in range(1, bnum+1):
        for x in range(1, bnum+1):
            future = dask.delayed(check_block)((z,y,x), output_path, nlevels, ch, samples_per_pixel)
            futures.append(future)
    with ProgressBar():
        dask_result = dask.compute(*futures)

    ret = True
    for b in dask_result:
        if not b:
            ret = False
            break
    return ret
    #dlist = [item for sublist in dask_result for item in sublist]

def gen_highest_resolution_blocks_from_slices(indirs: List[str], output_path: str, tmpdir_path: str, nlevels: int, task_num: int, maxbatch: int, ch: int, voxel_size_str: str, darray: da.Array, resume: bool):
    dim_leaf = darray.chunksize[:3]
    if darray.shape[2] >= 192:
        tiled_tif_conversion = True
        Path(tmpdir_path).mkdir(parents=True, exist_ok=True)
    else:
        tiled_tif_conversion = False
    
    for i in range(0, len(indirs)):
        img_files = [os.path.join(indirs[i], f) for f in os.listdir(indirs[i]) if f.endswith(('.tif', '.jp2'))]
        img_files.sort()

        logging.info("writing the highest level: channel " + str(ch+i))
        chunked_list = [img_files[j:j+dim_leaf[0]] for j in range(0, len(img_files), dim_leaf[0])]
        bnum = pow(2, nlevels - 1)
        for z in range(1, bnum+1):
            if resume and check_tiff_blocks(output_path=output_path, z=z, nlevels=nlevels, ch=ch, darray=darray):
                continue

            slice_files = chunked_list[z-1]
            if tiled_tif_conversion:
                slice_files = covert_tiff_to_tiled_tiff(input_paths=slice_files, task_num=task_num, outdir=tmpdir_path, maxbatch=maxbatch, resume=resume)

            save_tiff_blocks(input_slices=slice_files, output_path=output_path, z=z, nlevels=nlevels, task_num=task_num, maxbatch=maxbatch, ch=ch+i, voxel_size_str=voxel_size_str, darray=darray, resume=resume)

            if tiled_tif_conversion:
                delete_temporary_files(tmpdir=tmpdir_path, task_num=task_num, maxbatch=maxbatch)  
        logging.info("done")

def gen_highest_resolution_blocks_from_stack(infiles: List[str], output_path: str, nlevels: int, ch: int, darray: da.Array):
    dim = darray.shape[:3]
    dim_leaf = darray.chunksize[:3]
    for i in range(0, len(infiles)):
        images = dask_image.imread.imread(infiles[i])
        adjusted = images[:dim[0], :dim[1], :dim[2]]
        volume = adjusted.rechunk(dim_leaf)
        volume.map_blocks(save_block, output_path, nlevels, dim_leaf, ch+i, chunks=(1,1,1)).compute()

def downsample_octree_blocks(output_path: str, method: str, nlevels: int, task_num: int, maxbatch: int, ch_ids: List[int], voxel_size_str: str, darray: da.Array, ktxdir: str, delete_source: bool, resume: bool):
    dim_leaf = darray.chunksize[:3]

    dummy_img = np.zeros(dim_leaf, dtype=darray.dtype)
    cfpath = os.path.join(output_path, "dummy_generated")
    dummy_exists = False
    for ch in ch_ids:
        img_name = "default.{0}.tif".format(ch)
        full_path = os.path.join(output_path, img_name)
        if not resume or (resume and not os.path.exists(full_path)):
            skimage.io.imsave(full_path, dummy_img)
            dummy_exists = True
    if dummy_exists:
        with open(cfpath, 'w') as fp:
            pass

    for lv in range(nlevels-1, 0, -1):
        if lv == 1 and os.path.exists(cfpath):
            resume = False
        logging.info("downsampling level " + str(lv+1))
        futures = []
        bnum = pow(2, lv - 1)
        batch_block_num = (int)(bnum * bnum * bnum * len(ch_ids) / task_num)
        if maxbatch > 0 and batch_block_num > maxbatch:
            batch_block_num = maxbatch
        if batch_block_num < 1: 
            batch_block_num = 1
        coord_list = []
        vs = parse_voxel_size(voxel_size_str)
        vs = [r * 2 for r in vs]
        for z in range(1, bnum+1):
            for y in range(1, bnum+1):
                for x in range(1, bnum+1):
                    for c in ch_ids:
                        coord_list.append((z,y,x))
                        if len(coord_list) >= batch_block_num:
                            future = dask.delayed(downsample_and_save_block_batch)(coord_list, output_path, lv, dim_leaf, c, darray.dtype, method, vs, ktxdir, delete_source, resume)
                            futures.append(future)
                            coord_list = []
        if len(coord_list) > 0:
            future = dask.delayed(downsample_and_save_block_batch)(coord_list, output_path, lv, dim_leaf, c, darray.dtype, method, vs, ktxdir, delete_source, resume)
            futures.append(future)
        with ProgressBar():
            dask.compute(futures)
        logging.info("done")
    
    ktxroot = str(Path(ktxdir).parent)
    if ktxroot != output_path:
        try:
            for fpath in glob.glob(os.path.join(output_path, 'default.[0-9].tif')):
                fname = os.path.basename(fpath)
                copyfile(fpath, os.path.join(ktxroot, fname))
                os.remove(fpath)
        except Exception as err:
            logging.error(err)
            logging.error("Could not copy the lowest resolution tif images.")

    if os.path.exists(cfpath):
        os.remove(cfpath)


def ktx_conversion(indir: str, outdir: str, nlevels: int, task_num: int, maxbatch: int, delete_source: bool):
    if outdir != indir:
        ktx_mkdir = True
    else:
        ktx_mkdir = False
    for lv in range(nlevels, 0, -1):
        logging.info("ktx conversion level " + str(lv))
        futures = []
        bnum = pow(2, lv - 1)
        batch_block_num = (int)(bnum * bnum * bnum / task_num)
        if maxbatch > 0 and batch_block_num > maxbatch:
            batch_block_num = maxbatch
        if batch_block_num < 1: 
            batch_block_num = 1
        coord_list = []
        for z in range(1, bnum+1):
            for y in range(1, bnum+1):
                for x in range(1, bnum+1):
                    coord_list.append((z,y,x))
                    if len(coord_list) >= batch_block_num:
                        future_ktx = dask.delayed(convert_block_ktx_batch)(coord_list, indir, outdir, lv, True, True, ktx_mkdir if lv == nlevels else False, delete_source)
                        futures.append(future_ktx)
                        coord_list = []
        if len(coord_list) > 0:
            future_ktx = dask.delayed(convert_block_ktx_batch)(coord_list, indir, outdir, lv, True, True, ktx_mkdir if lv == nlevels else False, delete_source)
            futures.append(future_ktx)
        with ProgressBar():
            dask.compute(futures)
        logging.info("done")
    
    ktxroot = str(Path(outdir).parent)
    if ktxroot != indir:
        try:
            for fpath in glob.glob(os.path.join(indir, 'default.[0-9].tif')):
                fname = os.path.basename(fpath)
                copyfile(fpath, os.path.join(ktxroot, fname))
                os.remove(fpath)
        except Exception as err:
            logging.error(err)
            logging.error("Could not copy the lowest resolution tif images.")

def build_octree_from_tiff_slices():
    argv = sys.argv
    argv = argv[1:]

    #to do: try click
    usage_text = ("Usage:" + "  slice2octree.py" + " [options]")
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("-t", "--thread", dest="number", type=int, default=16, help="number of threads")
    parser.add_argument("-i", "--inputdir", dest="input", type=str, default="", help="input directories")
    parser.add_argument("-f", "--inputfile", dest="file", type=str, default="", help="input image stacks")
    parser.add_argument("-o", "--output", dest="output", type=str, default=None, help="output directory")
    parser.add_argument("-l", "--level", dest="level", type=int, default=1, help="number of levels")
    parser.add_argument("-c", "--channel", dest="channel", type=int, default=0, help="channel id")
    parser.add_argument("-d", "--downsample", dest="downsample", type=str, default='area', help="downsample method: 2ndmax, area, aa(anti-aliasing), spline")
    parser.add_argument("-m", "--monitor", dest="monitor", default=False, action="store_true", help="activate monitoring")
    parser.add_argument("--origin", dest="origin", type=str, default="0,0,0", help="position of the corner of the top-level image in nanometers")
    parser.add_argument("--voxsize", dest="voxsize", type=str, default="1.0,1.0,1.0", help="voxel size of the top-level image")
    parser.add_argument("--memory", dest="memory", type=str, default="16GB", help="memory amount per thread (for LSF cluster)")
    parser.add_argument("--project", dest="project", type=str, default=None, help="project name (for LSF cluster)")
    parser.add_argument("--maxjobs", dest="maxjobs", type=int, default=16, help="maximum jobs (for LSF cluster)")
    parser.add_argument("--walltime", dest="walltime", type=str, default="1:00", help="expected lifetime of a worker. Defaults to one hour, i.e. 1:00 (for LSF cluster)")
    parser.add_argument("--maxbatch", dest="maxbatch", type=int, default=0, help="number of blocks per job")
    parser.add_argument("--lsf", dest="lsf", default=False, action="store_true", help="use LSF cluster")
    parser.add_argument("--ktx", dest="ktx", default=False, action="store_true", help="generate ktx files")
    parser.add_argument("--ktxonly", dest="ktxonly", default=False, action="store_true", help="output only a ktx octree")
    parser.add_argument("--ktxout", dest="ktxout", type=str, default=None, help="output directory for a ktx octree")
    parser.add_argument("--cluster", dest="cluster", type=str, default=None, help="address of a dask scheduler server")
    parser.add_argument("--verbose", dest="verbose", default=False, action="store_true", help="enable verbose logging")
    parser.add_argument("--resume", dest="resume", default=False, action="store_true", help="resume processing")

    if not argv:
        parser.print_help()
        exit()

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    tnum = args.number
    indirs = args.input.split(",")
    infiles = args.file.split(",")
    outdir = args.output
    nlevels = args.level
    ch = args.channel
    dmethod = args.downsample
    monitoring = args.monitor
    ktxout = args.ktxout
    ktxonly = args.ktxonly
    ktx = args.ktx
    ktx_mkdir = False
    resume = args.resume

    if ktxout and not outdir:
        ktxonly = True

    if ktxout or ktxonly:
        ktx = True

    if ktx and not ktxout:
        ktxout = outdir

    if ktxonly:
        logging.info("output only ktx octree")
        if not ktxout:
            ktxout = outdir
        else:
            outdir = ktxout

    if ktx:
        ktxout = os.path.join(ktxout, "ktx")
    else:
        ktxout = None

    if ktxonly:
        outdir = ktxout
    
    if ktx and outdir != ktxout:
        ktx_mkdir = True

    tmpdir_name = "tmp"
    tmpdir = os.path.join(outdir, tmpdir_name)

    maxbatch = args.maxbatch
    
    client = setup_cluster(
        cluster_address=args.cluster,
        monitoring=monitoring,
        is_lsf=args.lsf,
        walltime=args.walltime,
        memory_limit=args.memory,
        lsf_maximum_jobs=args.maxjobs,
        thread_num=tnum,
        project=args.project
    )
    
    task_num = tnum * 2
    if client:
        tmp_batch_num = 0
        workers_info = client.scheduler_info()['workers']
        for k in workers_info.keys():
            tmp_batch_num += workers_info[k]['nthreads']
        if tmp_batch_num > 0:
            task_num = tmp_batch_num * 2

    logging.info("task_num: " + str(task_num))

    darray = None
    tiled_tif_conversion = False
    if len(indirs) > 0 and indirs[0]:
        darray = slice_to_dask_array(indirs[0], nlevels)
    elif len(infiles) > 0 and infiles[0]:
        darray = stack_to_dask_array(infiles[0], nlevels)
    else:
        logging.error('Please specify an input dataset.')
        return

    dim = darray.shape[:3]
    volume_dtype = darray
    samples_per_pixel = darray.shape[3]
    dim_leaf = darray.chunksize[:3]

    logging.info("Will generate octree with " + str(nlevels) +" levels to " + str(outdir))
    logging.info("Image dimensions: " + str(dim))
    logging.info("Image type: " + str(volume_dtype))
    logging.info("samples per pixel: " + str(samples_per_pixel))
    logging.info("Adjusted image size: " + str(dim) + ", Dim leaf: " + str(dim_leaf))
    logging.info("KTX directory: " + str(ktxout))

    cfpath = os.path.join(outdir, "do_step1")
    if not resume or (resume and not os.path.exists(os.path.join(outdir, "transform.txt"))):
        Path(outdir).mkdir(parents=True, exist_ok=True)
        with open(cfpath, 'w') as fp:
            pass

    save_transform_txt(outdir=outdir, origin=args.origin, voxsize=args.voxsize, nlevels=nlevels, ktxout=ktxout)

    if samples_per_pixel > 1:
        indirs = [indirs[0]]
        ch_ids = [ch+i for i in range(0, samples_per_pixel)]
    else:
        ch_ids = [ch+i for i in range(0, len(indirs))]

    if resume and not os.path.exists(cfpath):
        logging.info("Skipped gen_highest_resolution_blocks_from_slices.")
    else:
        #save the highest level
        if len(indirs) > 0: # image slices
            gen_highest_resolution_blocks_from_slices(indirs=indirs, output_path=outdir, tmpdir_path=tmpdir, nlevels=nlevels, task_num=task_num, maxbatch=maxbatch, ch=ch, voxel_size_str=args.voxsize, darray=darray, resume=resume)
        elif len(infiles) > 0: #tif stack
            gen_highest_resolution_blocks_from_stack(infiles=infiles, output_path=outdir, nlevels=nlevels, ch=ch, darray=darray)

    if os.path.exists(cfpath):
        os.remove(cfpath)

    downsample_octree_blocks(output_path=outdir, method=dmethod, nlevels=nlevels, task_num=task_num, maxbatch=maxbatch, ch_ids=ch_ids, voxel_size_str=args.voxsize, darray=darray, ktxdir=ktxout, delete_source=ktxonly, resume=resume)

    #if ktx:
    #    ktx_conversion(indir=outdir, outdir=ktxout, nlevels=nlevels, task_num=task_num, maxbatch=maxbatch, delete_source=ktxonly)

    try:
        if os.path.isdir(tmpdir):
            rmtree(tmpdir)
    except:
        logging.error("could not remove the temporary directory:" + tmpdir)

    if client:
        client.close()
    
    logging.info("build_octree_from_tiff_slices: Done.")


def main():
    build_octree_from_tiff_slices()


if __name__ == '__main__':
    main()
