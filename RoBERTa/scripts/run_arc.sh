#! /bin/bash

python3 main.py roberta-for-ranking-arc --do_train --do_test --seed 42 --epochs 16 --batch_size 8 --model_name roberta-large --learning_rate 1e-4

python3 main.py roberta-for-ranking-arc --do_train --do_test --seed 1128 --epochs 16 --batch_size 8 --model_name roberta-large --learning_rate 1e-4

python3 main.py roberta-for-ranking-arc --do_train --do_test --seed 1143 --epochs 16 --batch_size 8 --model_name roberta-large --learning_rate 1e-4

python3 main.py roberta-for-ranking-arc --do_train --do_test --seed 1385 --epochs 16 --batch_size 8 --model_name roberta-large --learning_rate 1e-4

python3 main.py roberta-for-ranking-arc --do_train --do_test --seed 1415 --epochs 16 --batch_size 8 --model_name roberta-large --learning_rate 1e-4