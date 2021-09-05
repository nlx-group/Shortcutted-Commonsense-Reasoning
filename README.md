# Shortcutted Commonsense: Data Spuriousness in Deep Learning of Commonsense Reasoning

This repository contains the code and some data (certain datasets cannot be shared due to licensing issues) for the [paper]():

```
Shortcutted Commonsense: Data Spuriousness in Deep Learning of Commonsense Reasoning
Ruben Branco, António Branco, João Silva and João Rodrigues
to appear at EMNLP 2021
```

## Abstract

> Commonsense is a quintessential human capacity that has been a core challenge to Artificial Intelligence since its inception. Impressive results in Natural Language Processing tasks, including in commonsense reasoning, have consistently been achieved with Transformer neural language models, even matching or surpassing human performance in some benchmarks. Recently, some of these advances have been called into question: so called data artifacts in the training data have been made evident as spurious correlations and shallow shortcuts that in some cases are leveraging these outstanding results.
>
> In this paper we seek to further pursue this analysis into the realm of commonsense related language processing tasks. We undertake a study on different prominent benchmarks that involve commonsense reasoning, along a number of key stress experiments, thus seeking to gain insight on whether the models are learning transferable generalizations intrinsic to the problem at stake or just taking advantage of incidental shortcuts in the data items.
>
> The results obtained indicate that most datasets experimented with are problematic, with models resorting to non-robust features and appearing not to be learning and generalizing towards the overall tasks intended to be conveyed or exemplified by the datasets.

## Code

The paper comprises six experiments in the pursuit of identifying Shortcut Learning when applying state-of-the-art Transformer models to Commonsense Reasoning tasks.

It follows a list of the experiments and the file associated with them:

- **Tasks baselines**: each models folder (`RoBERTa/`, `GPT-2/`, `T5/` and `BART_AND_COMET/`) contains a scripts folder, which provides an example of how to run the models for each task.

- **Partial Inputs**: To perform partial input training, modify the `data.py` file in each model folders by commenting the section of inputs to not be supplied. Afterwards, run the task training script again.

- **Adversarial Attack**: Adversarial attacks can be performed using the `create_adversarial.py` file in the `RoBERTa/` and `BART_AND_COMET/` folder. An example of how to run an adversarial attack is given in `RoBERTa/scripts/run_adversarial.sh`. This experiment required modifying the TextAttack library. The modified library can be found in `TextAttack/`. It is recommended to install this version of the library for reproduction purposes.

- **Data Contamination**: The code and requirements for the experiment is provided in the `Data_Contamination/` folder. The code and results for the experiment are found in a jupyter-notebook: `Data_Contamination/Contamination.ipynb`. The jupyter-notebook includes URLs to download the data that was not supplied in `data/`. Only one dataset cannot be obtained easily due to licensing issues, which is CC-News. The script `Data_Contamination/crawl_ccnews.py` can be used to retrieve the dataset, however, **it takes a LONG time**. You may contact rmbranco[at]fc[dot]ul[dot]pt for more details on how to obtain this dataset.

- **Cross Tasks**: The code for the cross tasks experiment is implemented in `BART_AND_COMET/cross_eval.py` and `RoBERTa/cross_eval.py`. A script to run this code is provided in `BART_AND_COMET/scripts/cross.sh` and `RoBERTa/scripts/cross.sh`.

- **Shortcut Exploration**: The Shortcut Exploration experiments, described in detail in the appendix, are implemented in the end of the data contamination jupyter-notebook (`Data_Contamination/Contamination.ipynb`).

## Citation

```
@inproceedings{branco-etal-2021-shortcutted-commonsense,
    title = "Shortcutted Commonsense: Data Spuriousness in Deep Learning of Commonsense Reasoning",
    author = "Branco, Ruben  and
        Branco, Ant{\'o}nio  and
        Silva, João and
        Rodrigues, João",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```
