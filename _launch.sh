#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=180:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=mdeq
#SBATCH --output=mdeq_job_%j.out

. /etc/profile.d/lmod.sh
. grandproj.env
module use /pkgs/environment-modules/
module load pytorch1.4-cuda10.1-python3.6
/h/skhalid/mdeq/_runner.sh
#(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &
#python /h/skhalid/pytorch.py
#wait
