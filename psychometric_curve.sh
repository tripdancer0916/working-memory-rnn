#!/usr/bin/env bash

for i in {0..21}; do
    for((j=`expr ${i} \* 2 + 1`;j<=`expr \( ${i} + 1 \) \* 2`;j++)); do
        echo ${j}
        nohup python -u psychometric_curve.py cfg/romo_config/${j}.cfg > log/psychometric_curve/${j}.log &
    done

    # 終了待ち
    wait

done


