#! /bin/bash

python3 main.py t5-for-arct --do_train --do_test --data_path ../data/arct/ --seed 42 --batch_size 8 --model_name t5-large --learning_rate 2e-5 --epochs 17

python3 main.py t5-for-arct --do_train --do_test --data_path ../data/arct/ --seed 1128 --batch_size 8 --model_name t5-large --learning_rate 2e-5 --epochs 17

python3 main.py t5-for-arct --do_train --do_test --data_path ../data/arct/ --seed 1143 --batch_size 8 --model_name t5-large --learning_rate 2e-5 --epochs 17

python3 main.py t5-for-arct --do_train --do_test --data_path ../data/arct/ --seed 1385 --batch_size 8 --model_name t5-large --learning_rate 2e-5 --epochs 17

python3 main.py t5-for-arct --do_train --do_test --data_path ../data/arct/ --seed 1415 --batch_size 8 --model_name t5-large --learning_rate 2e-5 --epochs 17