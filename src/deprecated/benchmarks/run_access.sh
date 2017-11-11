#!/bin/bash

args=("$@")
 mkdir ${args[1]}/access
 rm ${args[1]}/access/*.log # remove previous logs


for i in 1 2 4 6 8 10 25 
do
	for j in 1 6 12 18 24 30 36 42
	do
		echo taskset 0x1 ${args[0]}/access $[$i*1000000] $j
		taskset 0x1 ${args[0]}/access $[$i*1000000] $j ${args[1]}/access/access_$[i*    100000].log
	done
	
	echo " "
done


