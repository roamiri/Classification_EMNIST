#!/bin/bash
#Account and Email Information
#SBATCH -A RoohollahAmiri ## Kestral user ID
#SBATCH --mail-type=end
#SBATCH --mail-user=roohollahamiri@u.boisestate.edu
# Specify parition (queue)
#SBATCH --partition=batch
# Join output and errors into output.
#SBATCH -o chiasson.o%j
#SBATCH -e chiasson.e%j
# Specify job not to be rerunable.
#SBATCH --no-requeue
# Job Name.
#SBATCH --job-name="EMNIST_slurm"
# Specify walltime.
###SBATCH --time=24:00:00
# Specify number of requested nodes.
#SBATCH -N 1
# Specify the total number of requested procs:
#SBATCH -n 8
# Number of GPUs per node.
#SBATCH --gres=gpu:1
# load all necessary modules.
module load slurm/17.11.2
module load cuda90/toolkit/9.0.176
module load nvidia/optix/5.0.1
module load python/gcc/64/2.7 ## theano & tensorflow are also loaded.
# Echo commands to stdout (standard output).
set -x
# Copy your code & data to your Kestrel Home directory using
# the SFTP (secure file transfer protocol).
# Go to the directory where the actual BATCH file is present.
cd $HOME/Classification_EMNIST
# The “python” command runs your python code.
# All output is dumped into chiasson.o%j with “%j” replaced by your Job ID.
## The file Run_EMNIST_network2_cross_entropy.py must also
## be in $HOME/MichaelNielsen_code_chap3_EMNIS 
python test.py
