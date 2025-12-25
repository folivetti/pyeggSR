#!/bin/bash
for f in results/melanoma/*.csv; do
    # Get 2nd to last line -> extract last field -> append to file
    tail -n 2 "$f" | head -n 1 | awk '{print $4}'  >> training_scores.txt

    # Get last line -> extract 4th field -> append to file
    tail -n 1 "$f"  | awk '{print $4}' >> test_scores.txt
done
