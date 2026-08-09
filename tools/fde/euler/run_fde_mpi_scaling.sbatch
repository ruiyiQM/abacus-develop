#!/usr/bin/env bash
#SBATCH --job-name=fde-scale
#SBATCH --time=02:00:00
#SBATCH --export=NONE

set -Eeuo pipefail

layout=${1:?usage: run_fde_mpi_scaling.sbatch LAYOUT OMP_THREADS}
omp_threads=${2:?usage: run_fde_mpi_scaling.sbatch LAYOUT OMP_THREADS}
root=/cluster/home/zhourui/abacus-develop
template=${root}/test_abacus/fodft_parallel_validation/fodft_modes/openmpi_elpa
scratch=/cluster/scratch/zhourui/abacus-fde-mpi-scaling
case_dir=${scratch}/cases/${SLURM_JOB_ID}-${layout}

export PATH=/cluster/apps/slurm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /etc/environ.d/lmod.sh
export MODULEPATH=/cluster/software/lmods
source ${root}/euler/abacus_gcc_openmpi_env.sh

mkdir -p "${case_dir}"
cp "${template}/INPUT" "${template}/STRU" "${template}/KPT" \
    "${template}/FDE_CONFIG" "${case_dir}/"
sed -i \
    -e "s/^suffix[[:space:]].*/suffix                  fde_scaling_${layout}_${SLURM_JOB_ID}/" \
    -e 's/^scf_thr[[:space:]].*/scf_thr                 1.0e-12/' \
    -e 's/^scf_nmax[[:space:]].*/scf_nmax                1/' \
    -e 's/^ks_solver[[:space:]].*/ks_solver              genelpa/' \
    "${case_dir}/INPUT"

if (( omp_threads < 1 || omp_threads > SLURM_CPUS_PER_TASK )); then
    echo "OMP thread count must fit the allocated CPUs per rank" >&2
    exit 2
fi
export OMP_NUM_THREADS=${omp_threads}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
binary=${root}/build-gcc-openmpi/install/bin/abacus_std_para
source_commit=$(git -C "${root}" rev-parse HEAD)
binary_sha256=$(sha256sum "${binary}" | cut -d ' ' -f 1)

cd "${case_dir}"
/usr/bin/time -v srun --mpi=pmi2 --cpu-bind=cores --distribution=block:block \
    --ntasks="${SLURM_NTASKS}" --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    "${binary}" > abacus.out 2> abacus.err

python3 "${root}/tools/fde/euler/parse_fde_scaling.py" \
    --case-dir "${case_dir}" \
    --layout "${layout}" \
    --job-id "${SLURM_JOB_ID}" \
    --nodes "${SLURM_JOB_NUM_NODES}" \
    --ranks "${SLURM_NTASKS}" \
    --threads "${omp_threads}" \
    --allocated-cpus-per-rank "${SLURM_CPUS_PER_TASK}" \
    --source-commit "${source_commit}" \
    --binary-sha256 "${binary_sha256}" \
    > "${case_dir}/result.json"

cp "${case_dir}/result.json" "${scratch}/results/${layout}.json"
