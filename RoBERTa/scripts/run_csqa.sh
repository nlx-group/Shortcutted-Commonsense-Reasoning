#! /bin/bash

python3 main.py roberta-for-csqa --do_train --do_test --seed 42 --epochs 13 --batch_size 8 --model_name roberta-large --learning_rate 3e-4

python3 main.py roberta-for-csqa --do_train --do_test --seed 1128 --epochs 13 --batch_size 8 --model_name roberta-large --learning_rate 3e-4

python3 main.py roberta-for-csqa --do_train --do_test --seed 1143 --epochs 13 --batch_size 8 --model_name roberta-large --learning_rate 3e-4

python3 main.py roberta-for-csqa --do_train --do_test --seed 1385 --epochs 13 --batch_size 8 --model_name roberta-large --learning_rate 3e-4

python3 main.py roberta-for-csqa --do_train --do_test --seed 1415 --epochs 13 --batch_size 8 --model_name roberta-large --learning_rate 3e-4