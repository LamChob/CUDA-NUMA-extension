#!/bin/bash
mkdir bin
mkdir log

#make all

echo which benchmark should be run?
echo 1:	stencil benchmark
echo 2: access benchmark
echo 3: page migration benchmark 
read input

case $input in 
1)	make stencil		
	./benchmarks/run_stencil.sh ./bin ./log
	;;
2)	make access
	./benchmarks/run_access.sh ./bin ./log
	;;
3)	make migrate
	./benchmarks/run_page_migration.sh ./bin ./log
	;;
*)	echo please try again 
	;;

esac


