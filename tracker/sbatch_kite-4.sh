#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
##SBATCH -p pal-lab
##SBATCH --nodelist=g021,g024, g022, g023
##SBATCH -N4
#SBATCH -N1
##SBATCH --ntasks-per-node=1
#SBATCH -n1
#SBATCH -c 4
#SBATCH --mem 100G
#SBATCH --time 2-00:00:00
##SBATCH --reservation=pal-lab
#SBATCH --account=pal-lab
#SBATCH --gres gpu:1

## labels and outputs
#SBATCH --job-name=kite-4
#SBATCH --output=kite-%j.out  # %j is the unique jobID

## notifications
##SBATCH --mail-user=esdft@missouri.edu  # email address for notifications
##SBATCH --mail-type=ALL  # which type of notifications to send
#-------------------------------------------------------------------------------


echo "### Starting at: $(date) ###"

## Module Commands
##module load miniconda3
##module list

## Activate your Python Virtual Environment (if needed)
source activate dmaot

##cd /cluster/VAST/civalab/results/development_data_2024

cd /home/esdft/data/DMAOT-VOTS2023/tracker

python python_swinb_dm_deaot_integrate_SAM_hq_h_imad_59_visualize_save_gt_masks.py kite-4

##python python_swinb_dm_deaot_not_integrate_lvos.py 0
