# coding:utf-8

def run(*argv):
	import argparse
	import time
	from typing import List

	from utils import Storage

	parser = argparse.ArgumentParser(description='A language generation model with GPT2. Beamsearch,\
		dropout and batchnorm is supported.')
	args = Storage()

	parser.add_argument('--name', type=str, default=None,
		help='The name of your model, used for tensorboard, etc. Default: runXXXXXX_XXXXXX (initialized by current time)')
	parser.add_argument('--restore', type=str, default=None,
		help='Checkpoints name to load. \
			"NAME_last" for the last checkpoint of model named NAME. "NAME_best" means the best checkpoint. \
			You can also use "last" and "best", by default use last model you run. \
			Attention: "NAME_last" and "NAME_best" are not guaranteed to work when 2 models with same name run in the same time. \
			"last" and "best" are not guaranteed to work when 2 models run in the same time.\
			Default: None (don\'t load anything)')
	parser.add_argument('--mode', type=str, default="train",
		help='"train" or "test". Default: train')

	parser.add_argument('--pretrained_model', type=str, default="gpt2")
	parser.add_argument('--decode_mode', type=str, choices=['max', 'sample', 'gumbel', 'samplek', 'beam'], default='samplek',
		help='The decode strategy when freerun. Choices: max, sample, gumbel(=sample), \
			samplek(sample from topk), beam(beamsearch). Default: samplek')
	parser.add_argument('--top_k', type=int, default=10,
		help='The top_k when decode_mode == "beam" or "samplek"')
	parser.add_argument('--length_penalty', type=float, default=0.7,
		help='The beamsearch penalty for short sentences. The penalty will get larger when this becomes smaller.')
	parser.add_argument('--temperature', type=float, default=1,
		help='Temperature. Default: 1')

	parser.add_argument('--dataid', type=str, default='resources://MSCOCO',
		help='Directory for data set. Default: resources://MSCOCO')
	parser.add_argument('--convert_to_lower_letter', type=bool, default=True,
		help='Convert all tokens in dataset to lower case.')
	parser.add_argument('--epoch', type=int, default=100,
		help="Epoch for training. Default: 100")
	parser.add_argument('--batch_per_epoch', type=int, default=500,
		help="Batches per epoch. Default: 500")

	parser.add_argument('--out_dir', type=str, default="./output",
		help='Output directory for test output. Default: ./output')
	parser.add_argument('--log_dir', type=str, default="./tensorboard",
		help='Log directory for tensorboard. Default: ./tensorboard')
	parser.add_argument('--model_dir', type=str, default="./model",
		help='Checkpoints directory for model. Default: ./model')
	parser.add_argument('--cache_dir', type=str, default="./cache",
		help='Checkpoints directory for cache. Default: ./cache')
	parser.add_argument('--cpu', action="store_true",
		help='Use cpu.')
	parser.add_argument('--debug', action='store_true',
		help='Enter debug mode (using ptvsd).')
	parser.add_argument('--cache', action='store_true',
		help='Use cache for speeding up load data and wordvec. (It may cause problems when you switch dataset.)')
	parser.add_argument('--seed', type=int, default=0,
		help='Specify random seed. Default: 0')
	parser.add_argument('--lr', type=float, default=1e-4,
		help='Learning rate. Default: 0.0001')
	cargs = parser.parse_args(argv)

	# general setting
	args.name = cargs.name or time.strftime("run%Y%m%d_%H%M%S", time.localtime())
	args.restore = cargs.restore
	args.mode = cargs.mode
	args.out_dir = cargs.out_dir
	args.log_dir = cargs.log_dir
	args.model_dir = cargs.model_dir
	args.cache_dir = cargs.cache_dir
	args.debug = cargs.debug
	args.cache = cargs.cache
	args.cuda = not cargs.cpu

	## dataset settings
	args.dataid = cargs.dataid
	args.convert_to_lower_letter = cargs.convert_to_lower_letter

	## training settings
	args.epochs = cargs.epoch
	args.batch_per_epoch = cargs.batch_per_epoch
	args.lr = cargs.lr
	args.batch_size = 64
	args.batch_num_per_gradient = 4
	args.grad_clip = 5
	args.show_sample = [0]  # show which batch when evaluating at tensorboard
	args.max_sent_length = 50
	args.checkpoint_steps = 20
	args.checkpoint_max_to_keep = 5

	## arguments for restoring checkpoints
	args.restore_optimizer = True
	load_exclude_set: List[str] = []
	restoreCallback = None

	## architecture settings
	args.pretrained = "gpt2"  # the pretrained arguments for cotk.LanguageGeneration
	args.pretrained_model = cargs.pretrained_model  # the model from transformers

	## decoding settings
	args.decode_mode = cargs.decode_mode
	args.top_k = cargs.top_k
	args.length_penalty = cargs.length_penalty
	args.temperature = cargs.temperature

	## random seed
	args.seed = cargs.seed

	import random
	random.seed(cargs.seed)
	import torch
	torch.manual_seed(cargs.seed)
	import numpy as np
	np.random.seed(cargs.seed)

	from main import main

	main(args, load_exclude_set, restoreCallback)

if __name__ == '__main__':
	import sys
	run(*sys.argv[1:])
