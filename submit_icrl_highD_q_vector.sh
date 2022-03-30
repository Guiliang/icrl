#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=ICRL
task_name="train-Mojuco-ICRL"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate bak-galen-cr37
cd ./interface/
python train_commonroad_icrl.py icrl -p ICRL-FE2 --group HC-ICRL -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -bi 10 -ft 2e5 -ni 30 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5 -psis -ctkno 2.5 -nt 1 -d cuda -l "$log_dir"
