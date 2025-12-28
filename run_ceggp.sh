#!/bin/bash

START=$1
END=$2

for ((i=START; i<=END; i++));
do
	{ /usr/bin/time -f "Time: %E, Peak Memory: %M KB"  python ceggp.py >> results/melanoma/run_${i}.csv; } 2>> results/melanoma/time ;
done
