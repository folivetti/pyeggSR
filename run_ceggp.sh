#!/bin/bash

START=$1
END=$2

for ((i=START; i<=END; i++));
do
	time python ceggp.py >> results/melanoma/run_${i}.csv 2>> results/melanoma/time ;
done
