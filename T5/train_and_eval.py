from argparse import Namespace
import logging
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models import T5ForARCT, T5ForARC, T5ForPIQA, T5ForCSQA


LOG = logging.getLogger(__name__)


def train_and_eval_arct(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
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
    print_outputs,
):
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        t5 = T5ForARCT(
            hparams,
            data_path=data_path,
            epochs=epochs,
            print_outputs=print_outputs,
        )
    else:
        t5 = T5ForARCT.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            data_path=data_path,
            epochs=epochs,
            print_outputs=print_outputs,
        )
    t5_name = "T5-For-ARCT"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, t5_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=t5_name)
    kwargs = dict()

    if hparams.gradient_accumulation_steps:
        kwargs["accumulate_grad_batches"] = hparams.gradient_accumulation_steps

    if use_early_stop:
        kwargs["callbacks"] = [
            EarlyStopping(
                monitor=early_stop_metric,
                patience=early_stop_patience,
                mode=early_stop_mode,
            )
        ]

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=1,
        **kwargs
    )

    if do_train:
        LOG.info("Starting training")
        trainer.fit(t5)
    if do_test:
        LOG.info("Starting testing")
        trainer.test(t5)


def train_and_eval_arc(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
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
    print_outputs,
):
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        t5 = T5ForARC(
            hparams,
            epochs=epochs,
            print_outputs=print_outputs,
        )
    else:
        t5 = T5ForARC.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            epochs=epochs,
            print_outputs=print_outputs,
        )
    t5_name = "T5-For-ARC"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, t5_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=t5_name)
    kwargs = dict()

    if hparams.gradient_accumulation_steps:
        kwargs["accumulate_grad_batches"] = hparams.gradient_accumulation_steps

    if use_early_stop:
        kwargs["callbacks"] = [
            EarlyStopping(
                monitor=early_stop_metric,
                patience=early_stop_patience,
                mode=early_stop_mode,
            )
        ]

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=1,
        **kwargs
    )

    if do_train:
        LOG.info("Starting training")
        trainer.fit(t5)
    if do_test:
        LOG.info("Starting testing")
        trainer.test(t5)


def train_and_eval_piqa(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
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
    print_outputs,
):
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        t5 = T5ForPIQA(
            hparams,
            data_path=data_path,
            epochs=epochs,
            print_outputs=print_outputs,
        )
    else:
        t5 = T5ForPIQA.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            data_path=data_path,
            epochs=epochs,
            print_outputs=print_outputs,
        )
    t5_name = "T5-For-PIQA"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, t5_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=t5_name)
    kwargs = dict()

    if hparams.gradient_accumulation_steps:
        kwargs["accumulate_grad_batches"] = hparams.gradient_accumulation_steps

    if use_early_stop:
        kwargs["callbacks"] = [
            EarlyStopping(
                monitor=early_stop_metric,
                patience=early_stop_patience,
                mode=early_stop_mode,
            )
        ]

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=1,
        **kwargs
    )

    if do_train:
        LOG.info("Starting training")
        trainer.fit(t5)
    if do_test:
        LOG.info("Starting testing")
        trainer.test(t5)


def train_and_eval_csqa(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
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
    print_outputs,
):
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        t5 = T5ForCSQA(
            hparams,
            epochs=epochs,
            print_outputs=print_outputs,
        )
    else:
        t5 = T5ForCSQA.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            epochs=epochs,
            print_outputs=print_outputs,
        )
    t5_name = "T5-For-CSQA"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, t5_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=t5_name)
    kwargs = dict()

    if hparams.gradient_accumulation_steps:
        kwargs["accumulate_grad_batches"] = hparams.gradient_accumulation_steps

    if use_early_stop:
        kwargs["callbacks"] = [
            EarlyStopping(
                monitor=early_stop_metric,
                patience=early_stop_patience,
                mode=early_stop_mode,
            )
        ]

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=1,
        **kwargs
    )

    if do_train:
        LOG.info("Starting training")
        trainer.fit(t5)
    if do_test:
        LOG.info("Starting testing")
        trainer.test(t5)
