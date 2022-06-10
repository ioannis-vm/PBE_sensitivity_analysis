#!/bin/bash
# Job name:
#SBATCH --job-name=nlth_analysis
#
# Account:
#SBATCH --account=fc_hpc4pbee
#
# Partition:
#SBATCH --partition=savio
#
# Tasks per node
#SBATCH --ntasks-per-node=20
#
# Nodes
#SBATCH --nodes=8
#
# Wall clock limit:
#SBATCH --time=1-20:00:00
#
# Email Notification
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ioannis_vm@berkeley.edu
#
## Command(s) to run:
module load gcc openmpi # or module load intel openmpi
ht_helper.sh -t taskfile_office3
