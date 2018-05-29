#!/usr/bin/env sh

#SLA=$1
#RATE=$2

for RATE in 10 20 30 40 50 60 70 80 90 100
do 
    for SLA in 50 100 200
    do 
	echo "sla: $SLA, rate: $RATE"
        /bin/bash -c "exec python runapp1.py $SLA $RATE" &
        /bin/bash -c "exec python runapp2.py $SLA $RATE" &
        wait
    done
done
