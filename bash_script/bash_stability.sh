#!/usr/bin/env bash

nohup python -u trajectory_stability.py cfg/romo_config/1_var_1.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/1_var_2.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/1_var_3.cfg &

wait

nohup python -u trajectory_stability.py cfg/romo_config/1_var_4.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/1_var_5.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/1_var_6.cfg &

wait

nohup python -u trajectory_stability.py cfg/romo_config/6_var_1.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/6_var_2.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/6_var_3.cfg &

wait

nohup python -u trajectory_stability.py cfg/romo_config/6_var_4.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/6_var_5.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/6_var_6.cfg &

wait

nohup python -u trajectory_stability.py cfg/romo_config/11_var_1.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/11_var_2.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/11_var_3.cfg &

wait

nohup python -u trajectory_stability.py cfg/romo_config/11_var_4.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/11_var_5.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/11_var_6.cfg &

wait

nohup python -u trajectory_stability.py cfg/romo_config/12_var_1.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/12_var_2.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/12_var_3.cfg &

wait

nohup python -u trajectory_stability.py cfg/romo_config/12_var_4.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/12_var_5.cfg &
nohup python -u trajectory_stability.py cfg/romo_config/12_var_6.cfg &

wait

