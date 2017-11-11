#!/bin/bash
args=("$@")

mkdir ${args[1]}

for i in 1 2 4 6 8 10 100 1000 2000; do
	for j in 1 2 3 4 5 6 7 8; do 
		echo ${args[0]}/migrate $i $j
		${args[0]}/migrate $i $j>> ${args[1]}/${args[2]}
	done
	echo " "
done
