# tiff2octree
a tiff-to-octree converter for dask

## Initial Setup
1. Install miniconda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
2. Run ```conda install dask```
3. Run ```conda install dask_jobqueue```
4. Run ```conda install tifffile```
5. Run ```conda install skimage```

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
  
  --lsf                                   use LSF cluster
  --memory MEMORY                         memory amount per thread (for LSF cluster)
  --project PROJECT                       project name (for LSF cluster)
  --maxjobs MAXJOBS                       maximum jobs (for LSF cluster)
```
