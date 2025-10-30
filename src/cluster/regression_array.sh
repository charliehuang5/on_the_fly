#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output='logs/regression.%A.%a.log'

# Split up text file
text_file="$1"
fdaPath="$2"
linenum=$SLURM_ARRAY_TASK_ID
linenum=$((linenum - 1))
OLDIFS=$IFS
IFS=$'\n'
array=($(grep "[a-z]" $text_file))
linetxt=${array["$linenum"]}
datadir="$linetxt"
echo "$datadir"

# load module
module load anaconda3/2023.9
source activate brain_tools

# Export python path
PYTHONPATH='/tigress/MMURTHY/Max/courtship_dynamics':$PYTHONPATH
export PYTHONPATH

# Run script
currentDir=$(pwd)
scriptPath="$(dirname "$currentDir")/analysis/regression/run_regression.py"
python $scriptPath --dir $(dirname "$datadir")

# # Create movie
# echo "creating supervoxel movies."
# scriptPath="$(dirname "$currentDir")/analysis/regression/create_supervoxel_movie.py"
# python $scriptPath --dir $(dirname "$datadir") --fdaPath $fdaPath

# Create activation map
echo "creating activation map"
scriptPath="$(dirname "$currentDir")/analysis/regression/singlefly_regression_map.py"
python $scriptPath --dir $(dirname "$datadir") --fdaPath $fdaPath
