[![Main Repo](https://img.shields.io/badge/Main_project-cotk-blue.svg?logo=github)](https://github.com/thu-coai/cotk)
[![This Repo](https://img.shields.io/badge/Model_repo-GPT2LM--pytorch-blue.svg?logo=github)](https://github.com/thu-coai/GPT2LM-pytorch)
[![Coverage Status](https://coveralls.io/repos/github/thu-coai/GPT2LM-pytorch/badge.svg?branch=master)](https://coveralls.io/github/thu-coai/GPT2LM-pytorch?branch=master)
[![Build Status](https://travis-ci.com/thu-coai/GPT2LM-pytorch.svg?branch=master)](https://travis-ci.com/thu-coai/GPT2LM-pytorch)

# GPT2LM(PyTorch)

A language model by finetuning GPT2.

You can refer to the following paper for details:

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*, *1*(8), 9.

## Require Packages

* **python3**
* cotk >= 0.1.0
* pytorch >= 1.0.0
* tensorboardX >= 1.4
* transformers

## Quick Start

* Install ``CoTK`` following [instructions](https://github.com/thu-coai/cotk#installation).
* Using ``cotk download thu-coai/GPT2LM-pytorch/master`` to download codes.
* Execute ``python run.py`` to train the model.
  * The default dataset is ``resources://MSCOCO``. You can use ``--dataid`` to specify data path (can be a local path, a url or a resources id). For example: ``--dataid /path/to/datasets``
  * If you don't have GPUs, you can add `--cpu` for switching to CPU, but it may cost very long time for either training or test.
  * It may cost time downloading GPT2 pretrained model.
* You can view training process by tensorboard, the log is at `./tensorboard`.
  * For example, ``tensorboard --logdir=./tensorboard``. (You have to install tensorboard first.)
* After training, execute  ``python run.py --mode test --restore best`` for test.
  * You can use ``--restore filename`` to specify checkpoints files, which are in ``./model``. For example: ``--restore pretrained-mscoco`` for loading ``./model/pretrained-mscoco.model``
  * ``--restore last`` means last checkpoint, ``--restore best`` means best checkpoints on dev.
  * ``--restore NAME_last`` means last checkpoint with model named NAME. The same as``--restore NAME_best``.
* Find results at ``./output``.

## Example

WAIT FOR UPDATE

## Performance

WAIT FOR UPDATE

## Author

[HUANG Fei](https://github.com/hzhwcmhf)
