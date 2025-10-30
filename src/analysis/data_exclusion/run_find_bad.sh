#!/bin/bash
JOB_SCRIPT="find_bad_array.sh"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -d|--directory)
      DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--cpus)
      CPUS="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

# Paths to all experiments files (ignoring hidden dot files)
readarray EXP_DIRS < <(find $DIRECTORY -type f -path "*channel_1/*cleaned*.mmap")

ARRAY_ARGS_FILE="$DIRECTORY/array_args.txt"
rm -f "$ARRAY_ARGS_FILE"

for EXP_DIR in "${EXP_DIRS[@]}"
do
  NUM_ARRAY_JOBS=$[$NUM_ARRAY_JOBS +1]
  echo "$EXP_DIR" >> $ARRAY_ARGS_FILE
done

cat << EndOfMessage

Processing following directories (#jobs=$NUM_ARRAY_JOBS):

${EXP_DIRS[@]}"

EndOfMessage

# Run jobs
sbatch -n 1 -c "$CPUS" -a 1-"$NUM_ARRAY_JOBS" "$JOB_SCRIPT" "$ARRAY_ARGS_FILE" 
