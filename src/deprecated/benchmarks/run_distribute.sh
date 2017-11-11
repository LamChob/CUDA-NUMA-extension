#!/bin/bash
args=("$@")

for i in 2 4 6 8; 
do
	${args[0]}/migrate 10000000 0 $i
done

echo " "
