#!/bin/bash
# python MAB_Real.py --stop_rule SPRT --MAB_alg Elimination --total_user_each_arm 5000 --num_epoch 20
# python MAB_Real.py --stop_rule SPRT --MAB_alg UCB --total_user_each_arm 5000 --num_epoch 20
# python MAB_Real.py --stop_rule SPRT --MAB_alg TS --total_user_each_arm 3000 --num_epoch 20
python MAB_Real.py --stop_rule SPRT --MAB_alg US --total_user_each_arm 5000 --num_epoch 20
# python MAB_Real.py --stop_rule SPRT --MAB_alg EG --total_user_each_arm 3000 --num_epoch 20
python MAB_Real.py --stop_rule T-test --MAB_alg AB --total_user_each_arm 5000 --num_epoch 20
python MAB_Real.py --stop_rule PSI_Rule --MAB_alg PSI --total_user_each_arm 5000 --num_epoch 100
python MAB_Real.py --stop_rule esSR --MAB_alg esSR --total_user_each_arm 5000 --num_epoch 20
