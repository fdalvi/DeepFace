#!/bin/bash

if [ $# -ne 1 ]; then
    echo "./download_data.sh <path-to-set>"
    echo "Example: ./download_data.sh ../data/dev_set"
    echo "Example: ./download_data.sh ../data/eval_set"
    exit 1
fi

cd $1
mkdir -p images
cd images
while IFS=' ' read filename url junk junk2;
do
	wget -bqc -nc --read-timeout=10 -t 2 -O $filename $url;
done < ../links.txt
