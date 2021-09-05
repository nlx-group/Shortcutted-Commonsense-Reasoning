import os

import click

from train_and_eval import (
    train_and_eval_arct,
    train_and_eval_arc,
    train_and_eval_piqa,
    train_and_eval_csqa,
)

@click.group()
def cli():
    pass


@click.command()
@click.option("--do_train", is_flag=True, help="Train the model.")
@click.option("--do_test", is_flag=True, help="Test the model.")
@click.option(
    "--load_from_checkpoint",
    default=None,
    type=click.Path(),
    help="Whether to load the model from a given checkpoint. (default: None)",
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Epochs of training (default: 10)",
)
@click.option(
    "--learning_rate",
    default=1e-5,
    type=float,
    help="Learning rate (default: 1e-5)",
)
@click.option(
    "--weight_decay",
    default=0.1,
    type=float,
    help="Weight Decay for Adam opt (default: 0.1)",
)
@click.option(
    "--no_lr_schedule",
    is_flag=True,
    help="Don't use LR Scheduling.",
)
@click.option(
    "--warmup_ratio",
    default=0.06,
    type=float,
    help="Warmup ratio for linear schedule (default: 0.06)",
)
@click.option(
    "--gradient_accumulation_steps",
    default=0,
    type=int,
    help="How many batches should the gradient accumulate for before doing an update. (default: 0)",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed. (default: 42)",
)
@click.option(
    "--model_name",
    default="gpt2-base",
    type=str,
    help="Model name from huggingface gpt2 pre-trained models. (default: gpt2-base)",
)
@click.option(
    "--batch_size", default=8, type=int, help="Training batch size. (default: 8)"
)
@click.option(
    "--max_seq_len",
    default=102,
    type=int,
    help="Maximum length for any example sequence (in subwords). (default: 102)",
)
@click.option(
    "--data_path",
    default=os.path.join(os.getcwd(), "data"),
    type=click.Path(),
    help="Path to directory where data is located. (default: ./data/)",
)
@click.option(
    "--log_path",
    default=os.path.join(os.getcwd(), "logs"),
    type=click.Path(),
    help="Path to directory where tensorboard logs will be saved. (default: ./logs/)",
)
@click.option(
    "--checkpoint_path",
    default=os.path.join(os.getcwd(), "checkpoints"),
    type=click.Path(),
    help="Path to directory where model checkpoints will be saved. (default: ./checkpoints/)",
)
@click.option(
    "--save_top_k",
    default=-1,
    type=int,
    help=(
        "Save top k model checkpoints, if -1 saves all, else saves the input number, "
        "according to validation loss. (default: -1)"
    ),
)
@click.option(
    "--use_early_stop",
    is_flag=True,
    help="Whether to use early stopping.",
)
@click.option(
    "--early_stop_metric",
    type=str,
    default="Loss/Validation",
    help="Metric to monitor for early stopping criteria. (default: Loss/Validation)",
)
@click.option(
    "--early_stop_patience",
    type=int,
    default=3,
    help=(
        "Number of validation epochs with no improvement after which training will be "
        "stopped. (default: 3)"
    ),
)
@click.option(
    "--early_stop_mode",
    type=click.Choice(["auto", "min", "max"]),
    default="auto",
    help=(
        "Min mode, training will stop when the quantity monitored has stopped decreasing;"
        "in max mode it will stop when the quantity monitored has stopped increasing;"
    ),
)
def gpt2_for_arct(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    no_lr_schedule,
    warmup_ratio,
    gradient_accumulation_steps,
    seed,
    model_name,
    batch_size,
    max_seq_len,
    data_path,
    log_path,
    checkpoint_path,
    save_top_k,
    use_early_stop,
    early_stop_metric,
    early_stop_patience,
    early_stop_mode,
):
    train_and_eval_arct(
        do_train,
        do_test,
        load_from_checkpoint,
        epochs,
        learning_rate,
        weight_decay,
        not no_lr_schedule,
        warmup_ratio,
        gradient_accumulation_steps,
        seed,
        model_name,
        batch_size,
        max_seq_len,
        data_path,
        log_path,
        checkpoint_path,
        save_top_k,
        use_early_stop,
        early_stop_metric,
        early_stop_patience,
        early_stop_mode,
    )


@click.command()
@click.option("--do_train", is_flag=True, help="Train the model.")
@click.option("--do_test", is_flag=True, help="Test the model.")
@click.option(
    "--load_from_checkpoint",
    default=None,
    type=click.Path(),
    help="Whether to load the model from a given checkpoint. (default: None)",
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Epochs of training (default: 10)",
)
@click.option(
    "--learning_rate",
    default=1e-5,
    type=float,
    help="Learning rate (default: 1e-5)",
)
@click.option(
    "--weight_decay",
    default=0.1,
    type=float,
    help="Weight Decay for Adam opt (default: 0.1)",
)
@click.option(
    "--no_lr_schedule",
    is_flag=True,
    help="Don't use LR Scheduling.",
)
@click.option(
    "--warmup_ratio",
    default=0.06,
    type=float,
    help="Warmup ratio for linear schedule (default: 0.06)",
)
@click.option(
    "--gradient_accumulation_steps",
    default=0,
    type=int,
    help="How many batches should the gradient accumulate for before doing an update. (default: 0)",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed. (default: 42)",
)
@click.option(
    "--model_name",
    default="gpt2-base",
    type=str,
    help="Model name from huggingface gpt2 pre-trained models. (default: gpt2-base)",
)
@click.option(
    "--batch_size", default=8, type=int, help="Training batch size. (default: 8)"
)
@click.option(
    "--max_seq_len",
    default=90,
    type=int,
    help="Maximum length for any example sequence (in subwords). (default: 90)",
)
@click.option(
    "--log_path",
    default=os.path.join(os.getcwd(), "logs"),
    type=click.Path(),
    help="Path to directory where tensorboard logs will be saved. (default: ./logs/)",
)
@click.option(
    "--checkpoint_path",
    default=os.path.join(os.getcwd(), "checkpoints"),
    type=click.Path(),
    help="Path to directory where model checkpoints will be saved. (default: ./checkpoints/)",
)
@click.option(
    "--save_top_k",
    default=-1,
    type=int,
    help=(
        "Save top k model checkpoints, if -1 saves all, else saves the input number, "
        "according to validation loss. (default: -1)"
    ),
)
@click.option(
    "--use_early_stop",
    is_flag=True,
    help="Whether to use early stopping.",
)
@click.option(
    "--early_stop_metric",
    type=str,
    default="Loss/Validation",
    help="Metric to monitor for early stopping criteria. (default: Loss/Validation)",
)
@click.option(
    "--early_stop_patience",
    type=int,
    default=3,
    help=(
        "Number of validation epochs with no improvement after which training will be "
        "stopped. (default: 3)"
    ),
)
@click.option(
    "--early_stop_mode",
    type=click.Choice(["auto", "min", "max"]),
    default="auto",
    help=(
        "Min mode, training will stop when the quantity monitored has stopped decreasing;"
        "in max mode it will stop when the quantity monitored has stopped increasing;"
    ),
)
def gpt2_for_arc(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    no_lr_schedule,
    warmup_ratio,
    gradient_accumulation_steps,
    seed,
    model_name,
    batch_size,
    max_seq_len,
    log_path,
    checkpoint_path,
    save_top_k,
    use_early_stop,
    early_stop_metric,
    early_stop_patience,
    early_stop_mode,
):
    train_and_eval_arc(
        do_train,
        do_test,
        load_from_checkpoint,
        epochs,
        learning_rate,
        weight_decay,
        not no_lr_schedule,
        warmup_ratio,
        gradient_accumulation_steps,
        seed,
        model_name,
        batch_size,
        max_seq_len,
        log_path,
        checkpoint_path,
        save_top_k,
        use_early_stop,
        early_stop_metric,
        early_stop_patience,
        early_stop_mode,
    )


@click.command()
@click.option("--do_train", is_flag=True, help="Train the model.")
@click.option("--do_test", is_flag=True, help="Test the model.")
@click.option(
    "--load_from_checkpoint",
    default=None,
    type=click.Path(),
    help="Whether to load the model from a given checkpoint. (default: None)",
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Epochs of training (default: 10)",
)
@click.option(
    "--learning_rate",
    default=1e-5,
    type=float,
    help="Learning rate (default: 1e-5)",
)
@click.option(
    "--weight_decay",
    default=0.1,
    type=float,
    help="Weight Decay for Adam opt (default: 0.1)",
)
@click.option(
    "--no_lr_schedule",
    is_flag=True,
    help="Don't use LR Scheduling.",
)
@click.option(
    "--warmup_ratio",
    default=0.06,
    type=float,
    help="Warmup ratio for linear schedule (default: 0.06)",
)
@click.option(
    "--gradient_accumulation_steps",
    default=0,
    type=int,
    help="How many batches should the gradient accumulate for before doing an update. (default: 0)",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed. (default: 42)",
)
@click.option(
    "--model_name",
    default="gpt2-base",
    type=str,
    help="Model name from huggingface gpt2 pre-trained models. (default: gpt2-base)",
)
@click.option(
    "--batch_size", default=8, type=int, help="Training batch size. (default: 8)"
)
@click.option(
    "--max_seq_len",
    default=115,
    type=int,
    help="Maximum length for any example sequence (in subwords). (default: 115)",
)
@click.option(
    "--data_path",
    default=os.path.join(os.getcwd(), "data"),
    type=click.Path(),
    help="Path to directory where data is located. (default: ./data/)",
)
@click.option(
    "--log_path",
    default=os.path.join(os.getcwd(), "logs"),
    type=click.Path(),
    help="Path to directory where tensorboard logs will be saved. (default: ./logs/)",
)
@click.option(
    "--checkpoint_path",
    default=os.path.join(os.getcwd(), "checkpoints"),
    type=click.Path(),
    help="Path to directory where model checkpoints will be saved. (default: ./checkpoints/)",
)
@click.option(
    "--save_top_k",
    default=-1,
    type=int,
    help=(
        "Save top k model checkpoints, if -1 saves all, else saves the input number, "
        "according to validation loss. (default: -1)"
    ),
)
@click.option(
    "--use_early_stop",
    is_flag=True,
    help="Whether to use early stopping.",
)
@click.option(
    "--early_stop_metric",
    type=str,
    default="Loss/Validation",
    help="Metric to monitor for early stopping criteria. (default: Loss/Validation)",
)
@click.option(
    "--early_stop_patience",
    type=int,
    default=3,
    help=(
        "Number of validation epochs with no improvement after which training will be "
        "stopped. (default: 3)"
    ),
)
@click.option(
    "--early_stop_mode",
    type=click.Choice(["auto", "min", "max"]),
    default="auto",
    help=(
        "Min mode, training will stop when the quantity monitored has stopped decreasing;"
        "in max mode it will stop when the quantity monitored has stopped increasing;"
    ),
)
def gpt2_for_piqa(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    no_lr_schedule,
    warmup_ratio,
    gradient_accumulation_steps,
    seed,
    model_name,
    batch_size,
    max_seq_len,
    data_path,
    log_path,
    checkpoint_path,
    save_top_k,
    use_early_stop,
    early_stop_metric,
    early_stop_patience,
    early_stop_mode,
):
    train_and_eval_piqa(
        do_train,
        do_test,
        load_from_checkpoint,
        epochs,
        learning_rate,
        weight_decay,
        not no_lr_schedule,
        warmup_ratio,
        gradient_accumulation_steps,
        seed,
        model_name,
        batch_size,
        max_seq_len,
        data_path,
        log_path,
        checkpoint_path,
        save_top_k,
        use_early_stop,
        early_stop_metric,
        early_stop_patience,
        early_stop_mode,
    )


@click.command()
@click.option("--do_train", is_flag=True, help="Train the model.")
@click.option("--do_test", is_flag=True, help="Test the model.")
@click.option(
    "--load_from_checkpoint",
    default=None,
    type=click.Path(),
    help="Whether to load the model from a given checkpoint. (default: None)",
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Epochs of training (default: 10)",
)
@click.option(
    "--learning_rate",
    default=1e-5,
    type=float,
    help="Learning rate (default: 1e-5)",
)
@click.option(
    "--weight_decay",
    default=0.1,
    type=float,
    help="Weight Decay for Adam opt (default: 0.1)",
)
@click.option(
    "--no_lr_schedule",
    is_flag=True,
    help="Don't use LR Scheduling.",
)
@click.option(
    "--warmup_ratio",
    default=0.06,
    type=float,
    help="Warmup ratio for linear schedule (default: 0.06)",
)
@click.option(
    "--gradient_accumulation_steps",
    default=0,
    type=int,
    help="How many batches should the gradient accumulate for before doing an update. (default: 0)",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed. (default: 42)",
)
@click.option(
    "--model_name",
    default="gpt2-base",
    type=str,
    help="Model name from huggingface gpt2 pre-trained models. (default: gpt2-base)",
)
@click.option(
    "--batch_size", default=8, type=int, help="Training batch size. (default: 8)"
)
@click.option(
    "--max_seq_len",
    default=87,
    type=int,
    help="Maximum length for any example sequence (in subwords). (default: 87)",
)
@click.option(
    "--log_path",
    default=os.path.join(os.getcwd(), "logs"),
    type=click.Path(),
    help="Path to directory where tensorboard logs will be saved. (default: ./logs/)",
)
@click.option(
    "--checkpoint_path",
    default=os.path.join(os.getcwd(), "checkpoints"),
    type=click.Path(),
    help="Path to directory where model checkpoints will be saved. (default: ./checkpoints/)",
)
@click.option(
    "--save_top_k",
    default=-1,
    type=int,
    help=(
        "Save top k model checkpoints, if -1 saves all, else saves the input number, "
        "according to validation loss. (default: -1)"
    ),
)
@click.option(
    "--use_early_stop",
    is_flag=True,
    help="Whether to use early stopping.",
)
@click.option(
    "--early_stop_metric",
    type=str,
    default="Loss/Validation",
    help="Metric to monitor for early stopping criteria. (default: Loss/Validation)",
)
@click.option(
    "--early_stop_patience",
    type=int,
    default=3,
    help=(
        "Number of validation epochs with no improvement after which training will be "
        "stopped. (default: 3)"
    ),
)
@click.option(
    "--early_stop_mode",
    type=click.Choice(["auto", "min", "max"]),
    default="auto",
    help=(
        "Min mode, training will stop when the quantity monitored has stopped decreasing;"
        "in max mode it will stop when the quantity monitored has stopped increasing;"
    ),
)
def gpt2_for_csqa(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    no_lr_schedule,
    warmup_ratio,
    gradient_accumulation_steps,
    seed,
    model_name,
    batch_size,
    max_seq_len,
    log_path,
    checkpoint_path,
    save_top_k,
    use_early_stop,
    early_stop_metric,
    early_stop_patience,
    early_stop_mode,
):
    train_and_eval_csqa(
        do_train,
        do_test,
        load_from_checkpoint,
        epochs,
        learning_rate,
        weight_decay,
        not no_lr_schedule,
        warmup_ratio,
        gradient_accumulation_steps,
        seed,
        model_name,
        batch_size,
        max_seq_len,
        log_path,
        checkpoint_path,
        save_top_k,
        use_early_stop,
        early_stop_metric,
        early_stop_patience,
        early_stop_mode,
    )


if __name__ == "__main__":
    cli.add_command(gpt2_for_arct)
    cli.add_command(gpt2_for_arc)
    cli.add_command(gpt2_for_piqa)
    cli.add_command(gpt2_for_csqa)
    cli()