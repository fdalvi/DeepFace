#!/bin/bash

if [ $# -ne 1 ]; then
    echo "./crop_images.sh <path-to-set>"
    echo "Example: ./crop_images.sh ../data/dev_set"
    echo "Example: ./crop_images.sh ../data/eval_set"
    exit 1
fi

cd $1
mkdir -p images_cropped
cd images_cropped
while IFS=' ' read filename url rect md5; 
do 
	echo $filename $rect; 
	IFS=',' WORD_LIST=($rect);
	echo ${WORD_LIST[0]} ${WORD_LIST[1]} ${WORD_LIST[2]} ${WORD_LIST[3]};
	w="$(echo ${WORD_LIST[2]}-${WORD_LIST[0]} | bc)";
	h="$(echo ${WORD_LIST[3]}-${WORD_LIST[1]} | bc)";
	convert ../images/$filename -crop ${w}x${h}+${WORD_LIST[0]}+${WORD_LIST[1]} $filename;
	IFS=' ';
done < ../links.txt;
