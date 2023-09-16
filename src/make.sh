#!/bin/bash

python make.py --mode full --task_name s2s --run train --num_experiment 1 --round 8 --split_round 1
python make.py --mode full --task_name s2s --run test --num_experiment 1 --round 8 --split_round 1

python make.py --mode full --task_name clm --run train --num_experiment 1 --round 8 --split_round 1
python make.py --mode full --task_name clm --run test --num_experiment 1 --round 8 --split_round 1

python make.py --mode full --task_name sc --run train --num_experiment 1 --round 16 --split_round 1
python make.py --mode full --task_name sc --run test --num_experiment 1 --round 16 --split_round 1

python make.py --mode peft --task_name s2s --run train --num_experiment 1 --round 12 --split_round 1
python make.py --mode peft --task_name s2s --run test --num_experiment 1 --round 12 --split_round 1

python make.py --mode peft --task_name clm --run train --num_experiment 1 --round 8 --split_round 3
python make.py --mode peft --task_name clm --run test --num_experiment 1 --round 8 --split_round 3

python make.py --mode peft --task_name sc --run train --num_experiment 1 --round 24 --split_round 1
python make.py --mode peft --task_name sc --run test --num_experiment 1 --round 24 --split_round 1

python make.py --mode cola --task_name s2s --run train --num_experiment 1 --round 12 --split_round 1
python make.py --mode cola --task_name s2s --run test --num_experiment 1 --round 12 --split_round 1

python make.py --mode cola --task_name clm --run train --num_experiment 1 --round 8 --split_round 1
python make.py --mode cola --task_name clm --run test --num_experiment 1 --round 8 --split_round 1

python make.py --mode cola --task_name sc --run train --num_experiment 1 --round 24 --split_round 1
python make.py --mode cola --task_name sc --run test --num_experiment 1 --round 24 --split_round 1

python make.py --mode cola_step --task_name s2s --run train --num_experiment 1 --round 12 --split_round 1
python make.py --mode cola_step --task_name s2s --run test --num_experiment 1 --round 12 --split_round 1

python make.py --mode cola_step --task_name clm --run train --num_experiment 1 --round 8 --split_round 1
python make.py --mode cola_step --task_name clm --run test --num_experiment 1 --round 8 --split_round 1

python make.py --mode cola_step --task_name sc --run train --num_experiment 1 --round 24 --split_round 1
python make.py --mode cola_step --task_name sc --run test --num_experiment 1 --round 24 --split_round 1

python make.py --mode cola_dist --task_name clm --run train --num_experiment 1 --round 8 --split_round 1
python make.py --mode cola_dist --task_name clm --run test --num_experiment 1 --round 8 --split_round 1
