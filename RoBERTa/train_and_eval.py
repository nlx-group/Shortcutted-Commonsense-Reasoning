from argparse import Namespace
import logging
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models import (
    RobertaForARCT,
    RobertaForBinaryARCT,
    RobertaForPIQA,
    RobertaForMultipleChoicePIQA,
    RobertaForARC,
    RobertaForRankingARC,
    RobertaForCSQA,
)


LOG = logging.getLogger(__name__)


def train_and_eval_binary_arct(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    lr_schedule,
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
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        roberta = RobertaForBinaryARCT(
            hparams,
            data_path=data_path,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=2,
        )
    else:
        roberta = RobertaForBinaryARCT.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            data_path=data_path,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=2,
        )
    roberta_name = "Roberta-For-Binary-ARCT"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, roberta_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=roberta_name)
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
        trainer.fit(roberta)
    if do_test:
        LOG.info("Starting testing")
        trainer.test(roberta)


def train_and_eval_arct(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    lr_schedule,
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
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        roberta = RobertaForARCT(
            hparams,
            data_path=data_path,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=2,
        )
    else:
        roberta = RobertaForARCT.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            data_path=data_path,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=2,
        )
    roberta_name = "Roberta-For-ARCT"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, roberta_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=roberta_name)
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
        trainer.fit(roberta)
    if do_test:
        LOG.info("Starting testing")
        trainer.test(roberta)


def train_and_eval_piqa(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    lr_schedule,
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
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        roberta = RobertaForPIQA(
            hparams,
            data_path=data_path,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=2,
        )
    else:
        roberta = RobertaForPIQA.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            data_path=data_path,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=2,
        )
    roberta_name = "Roberta-For-PIQA"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, roberta_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=roberta_name)
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
        trainer.fit(roberta)
    if do_test:
        LOG.info("Starting testing")
        trainer.test(roberta)


def train_and_eval_mc_piqa(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    lr_schedule,
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
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        roberta = RobertaForMultipleChoicePIQA(
            hparams,
            data_path=data_path,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=2,
        )
    else:
        roberta = RobertaForMultipleChoicePIQA.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            data_path=data_path,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=2,
        )
    roberta_name = "Roberta-For-MC-PIQA"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, roberta_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=roberta_name)
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
        trainer.fit(roberta)
    if do_test:
        LOG.info("Starting testing")
        trainer.test(roberta)


def train_and_eval_arc(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    lr_schedule,
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
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        roberta = RobertaForARC(
            hparams,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=5,
        )
    else:
        roberta = RobertaForARC.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=5,
        )
    roberta_name = "Roberta-For-ARC"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, roberta_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=roberta_name)
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
        trainer.fit(roberta)
    if do_test:
        LOG.info("Starting testing")
        trainer.test(roberta)


def train_and_eval_ranking_arc(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    lr_schedule,
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
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        roberta = RobertaForRankingARC(
            hparams,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=5,
        )
    else:
        roberta = RobertaForRankingARC.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=5,
        )
    roberta_name = "Roberta-For-Ranking-ARC"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, roberta_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=roberta_name)
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
        trainer.fit(roberta)
    if do_test:
        LOG.info("Starting testing")
        trainer.test(roberta)


def train_and_eval_csqa(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    lr_schedule,
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
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        roberta = RobertaForCSQA(
            hparams,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=5,
        )
    else:
        roberta = RobertaForCSQA.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            epochs=epochs,
            lr_schedule=lr_schedule,
            num_classes=5,
        )
    roberta_name = "Roberta-For-CSQA"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, roberta_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=roberta_name)
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
        trainer.fit(roberta)
    if do_test:
        LOG.info("Starting testing")
        trainer.test(roberta)
