from argparse import Namespace
import click
from pytorch_lightning import Trainer

from models import (
    BARTRankingForARCT,
    BARTForARC,
    BARTForPIQA,
    BARTForCSQA,
)


TASK_TO_MODEL = dict(
    arct=BARTRankingForARCT,
    arc=BARTForARC,
    piqa=BARTForPIQA,
    csqa=BARTForCSQA,
)


def cross_evaluate(
    src,
    tgt,
    src_checkpoint,
    hparams,
    data_path,
    num_choices_src,
    num_choices_tgt,
):
    src_model = TASK_TO_MODEL[src].load_from_checkpoint(
        src_checkpoint,
        hparams=Namespace(**hparams),
        data_path=None,
        num_classes=num_choices_src,
    )
    tgt_model = TASK_TO_MODEL[tgt](
        hparams=Namespace(**hparams),
        data_path=data_path,
        init_pretrained_bart=False,
        num_classes=num_choices_tgt,
    )
    tgt_model.prepare_data()
    src_model.train_data = tgt_model.train_data
    src_model.val_data = tgt_model.val_data
    src_model.test_data = tgt_model.test_data

    trainer = Trainer(
        logger=False,
        checkpoint_callback=False,
        gpus=1,
    )
    print(f"RESULTS FOR MODEL TRAINED WITH {src.upper()} AND TESTED ON {tgt.upper()}")
    trainer.test(src_model)


@click.command()
@click.option("--src", type=click.Choice(list(TASK_TO_MODEL.keys())))
@click.option("--tgt", type=click.Choice(list(TASK_TO_MODEL.keys())))
@click.option(
    "--src_checkpoint",
    type=click.Path(),
)
@click.option(
    "--pretrained_weights",
    type=click.Path(),
)
@click.option(
    "--model_name",
    default="facebook/bart-large",
    type=str,
)
@click.option("--batch_size", type=int)
@click.option("--max_seq_len", type=int)
@click.option(
    "--data_path",
    type=click.Path(),
    default=None,
)
@click.option(
    "--num_choices_src",
    type=int,
)
@click.option(
    "--num_choices_tgt",
    type=int,
)
def evaluate(
    src,
    tgt,
    src_checkpoint,
    pretrained_weights,
    model_name,
    batch_size,
    max_seq_len,
    data_path,
    num_choices_src,
    num_choices_tgt,
):
    assert src != tgt
    cross_evaluate(
        src,
        tgt,
        src_checkpoint,
        dict(
            model_name=model_name,
            pretrained_weights=pretrained_weights,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            seed=1128,
        ),
        data_path,
        num_choices_src,
        num_choices_tgt,
    )


if __name__ == "__main__":
    evaluate()
