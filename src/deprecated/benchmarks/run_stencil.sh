#!/bin/bash
args=("$@")

mkdir ${args[1]}/stencil
rm ${args[1]}/stencil/*.log

for i in 1 2 4 8 16; do
	for j in 1 2 4 8; do
    		echo 	${args[0]}/stencil $[$i*1000] 499 127 $j 
		${args[0]}/stencil $[$i*1000] 499 127 $j >> ${args[1]}/stencil/stencil_$[$i*1000].log
	done
	
done
