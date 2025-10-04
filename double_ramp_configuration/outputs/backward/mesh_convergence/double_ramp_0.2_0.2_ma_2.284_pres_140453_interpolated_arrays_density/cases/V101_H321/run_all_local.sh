#!/bin/bash
# Auto-generated: sequential local execution of all SU2 cases
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo 'Running all local SU2 cases under:' $SCRIPT_DIR
START_TIME=$(date +%s)
COUNT=0
FAIL=0
for case_dir in $SCRIPT_DIR/*; do
  [ -d "$case_dir" ] || continue
  if [ -f "$case_dir/run.sh" ]; then
    echo "=== Case: $(basename $case_dir) ==="
    (cd "$case_dir" && ./run.sh) || { echo '  -> FAILED'; FAIL=$((FAIL+1)); }
    COUNT=$((COUNT+1))
  fi
done
END_TIME=$(date +%s)
echo "Completed $COUNT cases (failures: $FAIL) in $((END_TIME-START_TIME)) s"
