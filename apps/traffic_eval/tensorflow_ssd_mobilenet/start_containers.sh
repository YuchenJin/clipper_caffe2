#!/usr/bin/env sh

for i in `seq 1 10`;
do
  bash container_entry.sh tensorflow_container.py resnet$i 6 &
done    
