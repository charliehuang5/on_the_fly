#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=40000
##SBATCH --nodes=1
#SBATCH --output='logs/find_bad.%A.%a.log'

# plit up text file
text_file="$1"
linenum=$SLURM_ARRAY_TASK_ID
linenum=$((linenum-1))
OLDIFS=$IFS
IFS=$'\n'
array=( $(grep "[a-z]" $text_file) )
linetxt=${array["$linenum"]}
datadir="$linetxt"
echo "$datadir"

# Load conda environment
module load anaconda3/2020.2
source activate brain_tools

# Export python path
PYTHONPATH='/tigress/MMURTHY/Max/Brain_Tools':$PYTHONPATH
export PYTHONPATH

# Run script
scriptPath="find_bad_brains.py"
python $scriptPath --fname $datadir 
