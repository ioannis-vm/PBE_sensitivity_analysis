#!/bin/bash
# Job name:
#SBATCH --job-name=nlth_analysis_smrf_3_of_IV
#
# Account:
#SBATCH --account=fc_hpc4pbee
#
# Partition:
#SBATCH --partition=savio2_htc
#
# Tasks needed
#SBATCH --ntasks=40
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
ht_helper.sh -t savio/nlth_taskfile_smrf_3_of_IV
