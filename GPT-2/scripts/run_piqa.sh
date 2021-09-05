#! /bin/bash

python3 main.py gpt2-for-piqa --do_train --do_test --data_path ../data/PIQA/ --model_name gpt2-medium --learning_rate 1e-3 --batch_size 8 --epochs 22 --seed 42

python3 main.py gpt2-for-piqa --do_train --do_test --data_path ../data/PIQA/ --model_name gpt2-medium --learning_rate 1e-3 --batch_size 8 --epochs 22 --seed 1128

python3 main.py gpt2-for-piqa --do_train --do_test --data_path ../data/PIQA/ --model_name gpt2-medium --learning_rate 1e-3 --batch_size 8 --epochs 22 --seed 1143

python3 main.py gpt2-for-piqa --do_train --do_test --data_path ../data/PIQA/ --model_name gpt2-medium --learning_rate 1e-3 --batch_size 8 --epochs 22 --seed 1385

python3 main.py gpt2-for-piqa --do_train --do_test --data_path ../data/PIQA/ --model_name gpt2-medium --learning_rate 1e-3 --batch_size 8 --epochs 22 --seed 1415
