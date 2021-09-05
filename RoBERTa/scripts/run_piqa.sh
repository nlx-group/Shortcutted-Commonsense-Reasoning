#! /bin/bash

python3 main.py roberta-for-mc-piqa --do_train --do_test --seed 42 --epochs 15 --batch_size 4 --model_name roberta-large --learning_rate 1e-3 --data_path ../data/PIQA

python3 main.py roberta-for-mc-piqa --do_train --do_test --seed 1128 --epochs 15 --batch_size 4 --model_name roberta-large --learning_rate 1e-3 --data_path ../data/PIQA

python3 main.py roberta-for-mc-piqa --do_train --do_test --seed 1143 --epochs 15 --batch_size 4 --model_name roberta-large --learning_rate 1e-3 --data_path ../data/PIQA

python3 main.py roberta-for-mc-piqa --do_train --do_test --seed 1385 --epochs 15 --batch_size 4 --model_name roberta-large --learning_rate 1e-3 --data_path ../data/PIQA

python3 main.py roberta-for-mc-piqa --do_train --do_test --seed 1415 --epochs 15 --batch_size 4 --model_name roberta-large --learning_rate 1e-3 --data_path ../data/PIQA