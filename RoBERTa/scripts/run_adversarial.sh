# Example for CSQA

python3 create_adversarial.py --task csqa --method textfooler --model_name roberta-large --batch_size 8 --num_choices 5 --max_seq_len 90 --checkpoint_path <CSQA_CHECKPOINT_PATH> --save_dataset_path ./csqa-textfooler.csv --log_path ./csqa-textfooler.log