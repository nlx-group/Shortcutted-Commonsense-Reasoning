#! /bin/bash

python3 main.py roberta-for-arct --do_train --do_test --data_path ../data/arct/ --seed 42 --epochs 25 --batch_size 16 --model_name roberta-large

python3 main.py roberta-for-arct --do_train --do_test --data_path ../data/arct/ --seed 1128 --epochs 25 --batch_size 16 --model_name roberta-large

python3 main.py roberta-for-arct --do_train --do_test --data_path ../data/arct/ --seed 1143 --epochs 25 --batch_size 16 --model_name roberta-large

python3 main.py roberta-for-arct --do_train --do_test --data_path ../data/arct/ --seed 1385 --epochs 25 --batch_size 16 --model_name roberta-large

python3 main.py roberta-for-arct --do_train --do_test --data_path ../data/arct/ --seed 1415 --epochs 25 --batch_size 16 --model_name roberta-large