#! /bin/bash

python3 main.py bart-for-csqa --do_train --do_test --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 3e-4 --batch_size 8 --epochs 18 --seed 42

python3 main.py bart-for-csqa --do_train --do_test --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 3e-4 --batch_size 8 --epochs 18 --seed 1128

python3 main.py bart-for-csqa --do_train --do_test --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 3e-4 --batch_size 8 --epochs 18 --seed 1143

python3 main.py bart-for-csqa --do_train --do_test --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 3e-4 --batch_size 8 --epochs 18 --seed 1385

python3 main.py bart-for-csqa --do_train --do_test --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 3e-4 --batch_size 8 --epochs 18 --seed 1415
