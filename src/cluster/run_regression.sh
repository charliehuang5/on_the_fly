#!/bin/bash
JOB_SCRIPT="regression_array.sh"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -d|--directory)
      DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    -f|--fdaPath)
      FDAPATH="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

# Paths to all experiments files (ignoring hidden dot files)
readarray EXP_DIRS < <(find $DIRECTORY -name "*func" -type d -not -path "*fictrac")

ARRAY_ARGS_FILE="$DIRECTORY/array_args.txt"
rm -f "$ARRAY_ARGS_FILE"
for EXP_DIR in "${EXP_DIRS[@]}"
do
  NUM_ARRAY_JOBS=$[$NUM_ARRAY_JOBS +1]
  echo "$EXP_DIR" >> $ARRAY_ARGS_FILE
done

# Load module
module load anaconda3/2023.9
conda activate brain_tools

cat << EndOfMessage

Processing following directories (#jobs=$NUM_ARRAY_JOBS):

${EXP_DIRS[@]}"

EndOfMessage

# 1) Submit the array job and capture its job ID
job_id1=$(sbatch --array=1-"$NUM_ARRAY_JOBS" \
                "$JOB_SCRIPT" "$ARRAY_ARGS_FILE" "$FDAPATH" \
           | awk '{print $4}')

# # 2) Submit the second job (agglomeration) that depends on the first job
# job_id2=$(sbatch --dependency=afterany:$job_id1 <<EOF
# #!/bin/bash
# #SBATCH --job-name=agglomerate
# #SBATCH --output=logs/agglomerate_%j.log
# #SBATCH --mem=24G
# #SBATCH --time=02:00:00
# #SBATCH --cpus-per-task=1

# # Export python path
# PYTHONPATH='/tigress/MMURTHY/Max/courtship_dynamics':$PYTHONPATH
# export PYTHONPATH

# currentDir=$(pwd)
# scriptPath="$(dirname "$currentDir")/analysis/regression/agglomerate_h5s.py"
# python "$scriptPath" --dir "$DIRECTORY"
# EOF
# )

# # Capture the job ID of the second job 
# job_id2=$(echo "$job_id2" | awk '{print $4}')



