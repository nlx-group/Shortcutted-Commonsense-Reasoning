#! /bin/bash

python3 main.py gpt2-for-arc --do_train --do_test --model_name gpt2-medium --learning_rate 1e-3 --batch_size 4 --epochs 26 --seed 42

python3 main.py gpt2-for-arc --do_train --do_test --model_name gpt2-medium --learning_rate 1e-3 --batch_size 4 --epochs 26 --seed 1128

python3 main.py gpt2-for-arc --do_train --do_test --model_name gpt2-medium --learning_rate 1e-3 --batch_size 4 --epochs 26 --seed 1143

python3 main.py gpt2-for-arc --do_train --do_test --model_name gpt2-medium --learning_rate 1e-3 --batch_size 4 --epochs 26 --seed 1385

python3 main.py gpt2-for-arc --do_train --do_test --model_name gpt2-medium --learning_rate 1e-3 --batch_size 4 --epochs 26 --seed 1415
