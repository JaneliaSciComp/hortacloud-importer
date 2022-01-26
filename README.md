# tiff2octree
a tiff-to-octree converter for dask

## Initial Setup
1. Install miniconda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
2. Run ```conda install dask```
3. Run ```conda install dask-jobqueue -c conda-forge```
4. Run ```conda install dask-image```
5. Run ```conda install tifffile```
6. Run ```conda install scikit-image```
7. Run ```conda install -y libtiff```
8. Run ```conda install -y pyopengl```
9. Run ```conda install pylibtiff -c conda-forge```
10. Run ```conda install bitarray```

## Usage
```
commandline arguments:
  -h, --help                              show this help message and exit
  -n NUMBER, --thread NUMBER              number of threads (default: 16)
  -i INPUT, --inputdir INPUT              input directory
  -f FILE, --inputfile FILE               input image stack
  -o OUTPUT, --output OUTPUT              output directory
  -l LEVEL, --level LEVEL                 number of levels
  -c CHANNEL, --channel CHANNEL           channel id
  -d DOWNSAMPLE, --downsample DOWNSAMPLE  downsampling method: 2ndmax, area, aa (anti-aliasing)
  -m, --monitor                           activate monitoring. 
                                          you can see the dask dashboard at 
                                          http://(NodeName).int.janelia.org:8989/status
                                          (e.g. http://h07u01.int.janelia.org:8989/status)
  
  --origin ORIGIN       position of the corner of the top-level image in nanometers
  --voxsize VOXSIZE     voxel size of the top-level image
  --memory MEMORY       memory amount per thread (for LSF cluster)
  --project PROJECT     project name (for LSF cluster)
  --maxjobs MAXJOBS     maximum jobs (for LSF cluster)
  --lsf                 use LSF cluster
  --ktx                 generate ktx files
  --ktxout KTXOUT       output directory for a ktx octree
  
examples: 
python3 tiff2octree.py -i /input_slices/tiff -l 3 -o /output/octree -d area
python3 tiff2octree.py -i /input_slices/tiff -l 3 -o /output/octree -d area --lsf --project scicompsoft --memory 12GB --maxjobs 10
python3 tiff2octree.py -i /input_slices/ch1,/input_slices/ch2 -l 3 -o /output/octree --ktx --ktxout /output/octree/ktx
```
