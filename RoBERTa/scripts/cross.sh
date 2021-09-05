#! /bin/bash

python3 cross_eval.py --src arct --tgt arc --src_checkpoint <ARCT_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 90 --num_choices 5

python3 cross_eval.py --src arct --tgt piqa --src_checkpoint <ARCT_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 115 --num_choices 2 --data_path ../data/PIQA

python3 cross_eval.py --src arct --tgt csqa --src_checkpoint <ARCT_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 90 --num_choices 5

python3 cross_eval.py --src arc --tgt arct --src_checkpoint <ARC_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 80 --num_choices 2 --data_path ../data/arct

python3 cross_eval.py --src arc --tgt piqa --src_checkpoint <ARC_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 115 --num_choices 2 --data_path ../data/PIQA

python3 cross_eval.py --src arc --tgt csqa --src_checkpoint <ARC_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 90 --num_choices 5

python3 cross_eval.py --src piqa --tgt arct --src_checkpoint <PIQA_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 80 --num_choices 2 --data_path ../data/arct

python3 cross_eval.py --src piqa --tgt arc --src_checkpoint <PIQA_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 90 --num_choices 5

python3 cross_eval.py --src piqa --tgt csqa --src_checkpoint <PIQA_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 90 --num_choices 5

python3 cross_eval.py --src csqa --tgt arct --src_checkpoint <CSQA_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 80 --num_choices 2 --data_path ../data/arct

python3 cross_eval.py --src csqa --tgt arc --src_checkpoint <CSQA_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 90 --num_choices 5

python3 cross_eval.py --src csqa --tgt piqa --src_checkpoint <CSQA_MODEL_PATH> --model_name roberta-large --batch_size 8 --max_seq_len 115 --num_choices 2 --data_path ../data/PIQA