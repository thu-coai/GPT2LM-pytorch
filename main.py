# coding:utf-8
import os
import logging
import json

from cotk.dataloader import LanguageGeneration, PretrainedTokenizer
from transformers import GPT2Tokenizer

from utils import debug, try_cache, cuda_init, Storage
from gpt2lm import GPT2LM


def main(args, load_exclude_set, restoreCallback):
	logging.basicConfig(\
		filename=0,\
		level=logging.DEBUG,\
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',\
		datefmt='%H:%M:%S')

	if args.debug:
		debug()
	logging.info(json.dumps(args, indent=2))

	cuda_init(0, args.cuda)

	volatile = Storage()
	volatile.load_exclude_set = load_exclude_set
	volatile.restoreCallback = restoreCallback

	data_class = LanguageGeneration
	data_arg = Storage()
	data_arg.file_id = args.dataid
	data_arg.max_sent_length = args.max_sent_length
	data_arg.convert_to_lower_letter = args.convert_to_lower_letter
	data_arg.pretrained = args.pretrained
	data_arg.tokenizer = args.pretrained_model

	def load_dataset(data_arg):
		tokenizer = PretrainedTokenizer(GPT2Tokenizer.from_pretrained(data_arg.tokenizer))
		new_arg = Storage(data_arg.copy())
		new_arg.tokenizer = tokenizer
		dm = data_class(**new_arg)
		return dm

	if args.cache:
		dm = try_cache(load_dataset, (data_arg,),
			args.cache_dir, data_class.__name__)
	else:
		dm = load_dataset(data_arg)

	volatile.dm = dm

	param = Storage()
	param.args = args
	param.volatile = volatile

	model = GPT2LM(param)
	if args.mode == "train":
		model.train_process()
	elif args.mode == "test":
		test_res = model.test_process()

		json.dump(test_res, open("./result.json", "w"))
	else:
		raise ValueError("Unknown mode")
