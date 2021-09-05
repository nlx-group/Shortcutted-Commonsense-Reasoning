#! /bin/bash

python3 main.py bart-for-piqa --do_train --do_test --data_path ../data/PIQA/ --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 1e-3 --batch_size 4 --epochs 19 --seed 42

python3 main.py bart-for-piqa --do_train --do_test --data_path ../data/PIQA/ --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 1e-3 --batch_size 4 --epochs 19 --seed 1128

python3 main.py bart-for-piqa --do_train --do_test --data_path ../data/PIQA/ --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 1e-3 --batch_size 4 --epochs 19 --seed 1143

python3 main.py bart-for-piqa --do_train --do_test --data_path ../data/PIQA/ --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 1e-3 --batch_size 4 --epochs 19 --seed 1385

python3 main.py bart-for-piqa --do_train --do_test --data_path ../data/PIQA/ --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 1e-3 --batch_size 4 --epochs 19 --seed 1415
