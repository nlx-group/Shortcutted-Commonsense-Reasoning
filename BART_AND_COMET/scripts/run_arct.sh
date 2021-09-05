#! /bin/bash

python3 main.py bart-for-arct --do_train --do_test --data_path ../data/arct/ --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 1e-4 --batch_size 8 --epochs 25 --seed 42

python3 main.py bart-for-arct --do_train --do_test --data_path ../data/arct/ --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 1e-4 --batch_size 8 --epochs 25 --seed 1128

python3 main.py bart-for-arct --do_train --do_test --data_path ../data/arct/ --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 1e-4 --batch_size 8 --epochs 25 --seed 1143

python3 main.py bart-for-arct --do_train --do_test --data_path ../data/arct/ --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 1e-4 --batch_size 8 --epochs 25 --seed 1385

python3 main.py bart-for-arct --do_train --do_test --data_path ../data/arct/ --model_name facebook/bart-large --pretrained_weights <facebook/bart-large OR COMET_PRETRAIN_PATH> --learning_rate 1e-4 --batch_size 8 --epochs 25 --seed 1415
