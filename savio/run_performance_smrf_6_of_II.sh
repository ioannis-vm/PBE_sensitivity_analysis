#!/bin/bash
# Job name:
#SBATCH --job-name=si_smrf_6_of_II
#
# Account:
#SBATCH --account=fc_hpc4pbee
#
# Partition:
#SBATCH --partition=savio2_bigmem
#
# Tasks per node
#SBATCH --ntasks-per-node=24
#
# Nodes
#SBATCH --nodes=16
#
# Wall clock limit:
#SBATCH --time=20:00:00
#
# Email Notification
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ioannis_vm@berkeley.edu
#
## Command(s) to run:
module load gcc openmpi # or module load intel openmpi
ht_helper.sh -t savio/si_taskfile_smrf_6_of_II
