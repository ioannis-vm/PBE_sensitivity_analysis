#!/bin/bash
# Job name:
#SBATCH --job-name=design_smrf_6_of_II
#
# Account:
#SBATCH --account=fc_hpc4pbee
#
# Partition:
#SBATCH --partition=savio2
#
# Tasks needed
#SBATCH --ntasks=1
#
# Wall clock limit:
#SBATCH --time=1-20:00:00
#
# Email Notification
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ioannis_vm@berkeley.edu
#
## Command(s) to run:
module load python
source activate computing
python src/design_smrf_6_of_II.py
