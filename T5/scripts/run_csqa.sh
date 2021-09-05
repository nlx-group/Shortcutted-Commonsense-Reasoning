#! /bin/bash

python3 main.py t5-for-csqa --do_train --do_test --seed 42 --batch_size 8 --model_name t5-large --learning_rate 2e-5 --epochs 5

python3 main.py t5-for-csqa --do_train --do_test --seed 1128 --batch_size 8 --model_name t5-large --learning_rate 2e-5 --epochs 5

python3 main.py t5-for-csqa --do_train --do_test --seed 1143 --batch_size 8 --model_name t5-large --learning_rate 2e-5 --epochs 5

python3 main.py t5-for-csqa --do_train --do_test --seed 1385 --batch_size 8 --model_name t5-large --learning_rate 2e-5 --epochs 5

python3 main.py t5-for-csqa --do_train --do_test --seed 1415 --batch_size 8 --model_name t5-large --learning_rate 2e-5 --epochs 5