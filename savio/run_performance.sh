#!/bin/bash
# Job name:
#SBATCH --job-name=nlth_performance
#
# Account:
#SBATCH --account=fc_hpc4pbee
#
# Partition:
#SBATCH --partition=savio_bigmem
#
# Tasks per node
#SBATCH --ntasks-per-node=20
#
# Nodes
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=1-20:00:00
#
# Email Notification
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ioannis_vm@berkeley.edu
#
## Command(s) to run:
module load python && source activate computing && make -j$(nproc) make/performance
