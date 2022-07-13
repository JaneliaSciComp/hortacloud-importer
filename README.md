# hortacloud-importer
a tiff-to-octree converter for Horta.

## Initial Setup
1. Install miniconda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
2. Clone pyktx (https://github.com/JaneliaSciComp/pyktx)
3. Run ```conda env create -f environment.yml```
4. Run ```conda activate octree```
5. Run ```pip install /path/to/pyktx```
6. Run ```conda config --set auto_activate_base false``` (for LSF cluster)


## Convert Tiff Slices on a Local PC
1. Create an input folder for putting your tiff slices.
2. Copy your tiff slices to the input folder.
3. Create an output folder.
4. Run the following command.
```
conda activate octree
python tiff2octree.py -i /input_slices/tiff -o /output/octree -d 2ndmax -t 16 --ktx
```
This command generates both tiff and ktx octrees.  
```
-i: set path to your input folder.  
-o: set path to your output folder.  
-d: downsampling method. you can use 2ndmax, area, aa (anti-aliasing), spline. (2ndmax is being used for the mousdlight project.)  
-t: thread number.  
--ktx: generate a ktx compressed octree. You need to generate a KTX octree for browsing your data on Horta3D viewer. By default, this converter generates only a tiff octree.  
```
This converter aoutomatically determine the optimal number of levels for your data if you do not set the number of levels by using -l option.

If you browse your data only on Horta3D, please use --ktxonly option. The converter will generate only a ktx octree without a tiff octree.
```
python tiff2octree.py -i /input_slices/tiff -o /output/octree -d 2ndmax -t 16 --ktxonly
```

You can convert a multi-channel image by the following command. 
```
python tiff2octree.py -i /input_slices/ch1,/input_slices/ch2 -o /output/octree/ -d 2ndmax -t 8 --ktx
```
You need to create multiple folders for input data. (e.g. /input_slices/ch1, /input_slices/ch2)

You can load your local octree data to Janelia Workstation by File > New > Tiled Microscope Sample.

## Convert a Tiff Stack on a Local PC
The datasize of your tiff stack must be smaller than system memory.
1. Create an output folder.
2. Run the following command.
```
conda activate octree
python tiff2octree.py -f /input_slices/tiff -o /output/octree -d 2ndmax -t 16 --ktx
```
```
-f: set path to your input tif stack.
-o: set path to your output folder.
-d: downsampling method. you can use 2ndmax, area, aa (anti-aliasing), spline. (2ndmax is being used for the mousdlight project.)
-t: thread number. 
--ktx: generate a ktx compressed octree. You need to generate a KTX octree for browsing your data on Horta3D viewer. By default, this converter generates only a tiff octree.
```
You must use -f option for setting your tif stack as input.


## Convert Tiff Slices on the Janelia LSF cluster

1. Create an input folder for putting your tiff slices.
2. Copy your tiff slices to the input folder.
3. Create an output folder.
4. Run the following command.
```
conda activate octree
bsub -n 1 -W 24:00 -o log_output.txt -P scicompsoft "python tiff2octree.py -i /input_slices/tiff -o /output/octree -d 2ndmax -t 10 --ktx --lsf --project scicompsoft --memory 16GB --walltime 8:00"
```
```
-i: set path to your input folder.
-o: set path to your output folder.
-d: downsampling method. you can use 2ndmax, area, aa (anti-aliasing), spline. (2ndmax is being used for the mousdlight project.)
-t: thread number.
--ktx: generate a ktx compressed octree. You need to generate a KTX octree for browsing your data on Horta3D viewer. By default, this converter generates only a tiff octree.
--lsf: this option is necessary to use the lsf cluster.
--project: set a project name to be charged the cost for the janelia lsf cluster.
--memory: amount of memory per thread.
--walltime: runtime limit of each job. The default runtime limit is 1:00. If you are trying to convert large data, you may need to set a longer time limit.
```

## Resume a Stopped Process
If a process is terminated in the middle of execution, you can resume it by using ```--resume``` option.

if the following process is stopped in the middle of execution:
```
conda activate octree
bsub -n 1 -W 24:00 -o log_output.txt -P scicompsoft "python tiff2octree.py -i /input_slices/tiff -o /output/octree -d 2ndmax -t 10 --ktx --lsf --project scicompsoft --memory 16GB --walltime 8:00"
```

You can resume the process by the following command:
```
conda activate octree
bsub -n 1 -W 24:00 -o log_output.txt -P scicompsoft "python tiff2octree.py -i /input_slices/tiff -o /output/octree -d 2ndmax -t 10 --ktx --lsf --project scicompsoft --memory 16GB --walltime 8:00 --resume"
```

## Usage
```
commandline arguments:
  -h, --help                              show this help message and exit
  -t NUMBER, --thread NUMBER              number of threads (default: 16)
  -i INPUT, --inputdir INPUT              input directory
  -f FILE, --inputfile FILE               input image stack
  -o OUTPUT, --output OUTPUT              output directory
  -l LEVEL, --level LEVEL                 number of levels
  -c CHANNEL, --channel CHANNEL           channel id
  -d DOWNSAMPLE, --downsample DOWNSAMPLE  downsampling method: 2ndmax, area, aa (anti-aliasing), spline
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
  --cluster CLUSTER     address of a dask scheduler server
  --verbose             enable vorbose logging
  --resume              resume processing


examples: 

1. use a local cluster. (process image slices)
conda activate octree
python tiff2octree.py -i /input_slices/tiff -l 3 -o /output/octree -d 2ndmax -t 16

2. use a local cluster. (process image stack)
conda activate octree
python tiff2octree.py -f /path/to/tiff_stack.tif -l 3 -o /output/octree -d 2ndmax -t 16

3. use a LSF cluster.
conda activate octree
python tiff2octree.py -i /input_slices/tiff -l 3 -o /output/octree -d 2ndmax --lsf --project scicompsoft --memory 12GB --maxjobs 10 -t 10

4. output a ktx octree without a tiff octree.
conda activate octree
python tiff2octree.py -i /input_slices/ch1,/input_slices/ch2 -l 3 -o /output/octree/ -ktxonly -d 2ndmax -t 8

5. specify a cluster by its address.
conda activate octree
python tiff2octree.py -i /input_slices/tiff -l 3 -o /output/octree --cluster tcp://10.60.0.223:8786 -d spline -t 16
```
