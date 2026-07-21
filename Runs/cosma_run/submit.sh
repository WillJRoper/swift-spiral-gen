#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH -J LUXA-Spiral
#SBATCH --output=logs/luxa_spiral%j.log
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBATCH --time=72:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

module purge
module load intel_comp/2024.2.0 compiler-rt tbb compiler mpi ucx/1.17.0 parallel_hdf5/1.14.4 fftw/3.3.10 parmetis/4.0.3-64bit gsl/2.8

export SWIFT_BIN="${SWIFT_BIN:-$MSWIFTSIM/swift}"
export THREADS="${THREADS:-$SLURM_CPUS_PER_TASK}"
export MAKE_MOVIE="${MAKE_MOVIE:-0}"

./run.sh

echo "Job done, info follows..."
sacct -j "$SLURM_JOBID" --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode
exit
