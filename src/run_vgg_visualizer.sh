#!/bin/bash

for i in `seq 1 10`;
do
	end_idx=$(echo $i*900 | bc)
	start_idx=$(echo $end_idx-900 | bc)

	python vgg_visualizer.py $start_idx $end_idx &
done