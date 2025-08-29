#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=mpiTest
#SBATCH --ntasks=5
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:05:00

module load intel/oneAPI
module load anaconda
source activate causal-inf-env

echo "Python being used: $(which python)"
echo "MPI being used: $(which mpirun)"

echo "Testing numpy..."
python -c "import numpy; print(numpy.__version__)"
echo "Testing mpi4py..."
python -c "import mpi4py; print(mpi4py.__version__)"

echo "-------------------------------------------------------------"
echo "Testing a very short evaluation run"
mpirun -n 5 python Evaluation.py
