#!/bin/bash
#
#SBATCH --job-name=ftt
#SBATCH --output=./ftt.out
#
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=mp-capture01
#SBATCH --error=./ftt.err

./ftt 536870912
