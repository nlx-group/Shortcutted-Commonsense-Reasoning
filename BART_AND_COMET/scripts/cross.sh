#! /bin/bash

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src arct --num_choices_src 2 --tgt arc --src_checkpoint <COMET_ARCT_PATH> --batch_size 8 --max_seq_len 91 --num_choices_tgt 5

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src arct --num_choices_src 2 --tgt piqa --src_checkpoint <COMET_ARCT_PATH> --batch_size 8 --max_seq_len 101 --num_choices_tgt 2 --data_path ../data/PIQA/

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src arct --num_choices_src 2 --tgt csqa --src_checkpoint <COMET_ARCT_PATH> --batch_size 8 --max_seq_len 88 --num_choices_tgt 5

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src arc --num_choices_src 5 --tgt arct --src_checkpoint <COMET_ARC_PATH> --batch_size 8 --max_seq_len 101 --num_choices_tgt 2 --data_path ../data/arct/

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src arc --num_choices_src 5 --tgt piqa --src_checkpoint <COMET_ARC_PATH> --batch_size 8 --max_seq_len 101 --num_choices_tgt 2 --data_path ../data/PIQA/

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src arc --num_choices_src 5 --tgt csqa --src_checkpoint <COMET_ARC_PATH> --batch_size 8 --max_seq_len 88 --num_choices_tgt 5

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src piqa --num_choices_src 2 --tgt arct --src_checkpoint <COMET_PIQA_PATH> --batch_size 8 --max_seq_len 101 --num_choices_tgt 2 --data_path ../data/arct/

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src piqa --num_choices_src 2 --tgt arc --src_checkpoint <COMET_PIQA_PATH> --batch_size 8 --max_seq_len 91 --num_choices_tgt 5

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src piqa --num_choices_src 2 --tgt csqa --src_checkpoint <COMET_PIQA_PATH> --batch_size 8 --max_seq_len 88 --num_choices_tgt 5

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src csqa --num_choices_src 5 --tgt arct --src_checkpoint <COMET_CSQA_PATH> --batch_size 8 --max_seq_len 101 --num_choices_tgt 2 --data_path ../data/arct/

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src csqa --num_choices_src 5 --tgt arc --src_checkpoint <COMET_CSQA_PATH> --batch_size 8 --max_seq_len 88 --num_choices_tgt 5

python3 cross_eval.py --pretrained_weights <COMET_PRETRAIN_WEIGHTS_PATH> --src csqa --num_choices_src 5 --tgt piqa --src_checkpoint <COMET_CSQA_PATH> --batch_size 8 --max_seq_len 101 --num_choices_tgt 2 --data_path ../data/PIQA/