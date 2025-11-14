#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=causal-inference-eval
#SBATCH --account=dominoes
#SBATCH --ntasks=99
#SBATCH --cpus-per-task=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=02:00:00

module load intel/oneAPI
module load anaconda
eval "$(conda shell.bash hook)"
source activate causal-inf-env

export NUMBA_DISABLE_CACHE=0
export NUMBA_CACHE_DIR=/p/tmp/niklaslo/numba_cache
mkdir -p "$NUMBA_CACHE_DIR"

echo "Python being used: $(which python)"
echo "MPI being used: $(which mpirun)"

echo "Testing numpy..."
python -c "import numpy; print(numpy.__version__)"
echo "Testing mpi4py..."
python -c "import mpi4py; print(mpi4py.__version__)"

echo "-------------------------------------------------------------"
echo "Full Evaluation run"
mpirun -n 99 python Evaluation.py
