#!/bin/bash

START=$1
END=$2

mkdir -p results/$3

for ((i=START; i<=END; i++));
do
	{ /usr/bin/time -f "Time: %E, Peak Memory: %M KB"  python ceggp.py $3 >> results/${3}/run_${i}.csv; } 2>> results/${3}/time ;
done
