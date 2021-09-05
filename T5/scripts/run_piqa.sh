#! /bin/bash

python3 main.py t5-for-piqa --do_train --do_test --data_path ../data/PIQA --seed 42 --batch_size 8 --model_name t5-large --learning_rate 1e-5 --epochs 9

python3 main.py t5-for-piqa --do_train --do_test --data_path ../data/PIQA --seed 1128 --batch_size 8 --model_name t5-large --learning_rate 1e-5 --epochs 9

python3 main.py t5-for-piqa --do_train --do_test --data_path ../data/PIQA --seed 1143 --batch_size 8 --model_name t5-large --learning_rate 1e-5 --epochs 9

python3 main.py t5-for-piqa --do_train --do_test --data_path ../data/PIQA --seed 1385 --batch_size 8 --model_name t5-large --learning_rate 1e-5 --epochs 9

python3 main.py t5-for-piqa --do_train --do_test --data_path ../data/PIQA --seed 1415 --batch_size 8 --model_name t5-large --learning_rate 1e-5 --epochs 9