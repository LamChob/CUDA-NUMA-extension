#!/bin/bash
#
#SBATCH --job-name=managed-test
#SBATCH --output=./managed.out
#
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=mp-capture01
#SBATCH --error=./managed.err

./managed-test 16384
