# coding:utf-8
import logging

import torch
from torch import nn

from utils import zeros, LongTensor,\
			BaseNetwork, MyGRU, Storage, gumbel_max, flattenSequence
from utils.gpt2_helper import GPT2Decoder

# pylint: disable=W0221
class Network(BaseNetwork):
	def __init__(self, param):
		super().__init__(param)
		self.genNetwork = GenNetwork(param)

	def forward(self, incoming):
		incoming.result = Storage()
		self.genNetwork.forward(incoming)
		incoming.result.loss = incoming.result.word_loss
		if torch.isnan(incoming.result.loss).detach().cpu().numpy() > 0:
			logging.info("Nan detected")
			logging.info(incoming.result)
			raise FloatingPointError("Nan detected")

	def detail_forward(self, incoming):
		incoming.result = Storage()
		self.genNetwork.detail_forward(incoming)

class GenNetwork(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param

		self.decoder = GPT2Decoder(args.pretrained_model)
		self.start_generate_id = param.volatile.dm.eos_id
		self.lossCE = nn.CrossEntropyLoss()

	def teacherForcing(self, inp, gen):
		sent = inp.sent
		attn_mask = inp.attn_mask
		gen.w = self.decoder.forward(sent, attn_mask)

	def freerun(self, inp, gen):
		#mode: beam = beamsearch; max = choose max; sample = random_sampling; sample10 = sample from max 10

		if self.args.decode_mode == "beam":
			new_gen = self.decoder.beamsearch(inp, self.args.top_k, length_penalty=self.args.length_penalty)
			w_o = []
			length = []
			for i in range(inp.batch_size):
				w_o.append(new_gen.w_o[:, i, 0])
				length.append(new_gen.length[i][0])
			gen.w_o = torch.stack(w_o).transpose(0, 1)
			gen.length = length

		else:
			new_gen = self.decoder.freerun(inp, self.args.decode_mode, top_k=self.args.top_k, temperature=self.args.temperature)
			gen.w_o = new_gen.w_o
			gen.length = new_gen.length

	def forward(self, incoming):
		inp = Storage()
		inp.attn_mask = incoming.data.sent_attnmask
		inp.sent = incoming.data.sent

		incoming.gen = gen = Storage()
		self.teacherForcing(inp, gen)

		w_o_f = flattenSequence(gen.w.transpose(0, 1), incoming.data.sent_length-1)
		data_f = flattenSequence(incoming.data.sent.transpose(0, 1)[1:], incoming.data.sent_length-1)
		incoming.result.word_loss = self.lossCE(w_o_f, data_f)
		incoming.result.perplexity = torch.exp(incoming.result.word_loss)

	def detail_forward(self, incoming):
		inp = Storage()
		batch_size = inp.batch_size = incoming.data.batch_size
		inp.dm = self.param.volatile.dm
		inp.max_sent_length = self.args.max_sent_length
		inp.go_id = inp.dm.eos_id
		inp.eos_id = inp.dm.eos_id

		incoming.gen = gen = Storage()
		self.freerun(inp, gen)

		dm = self.param.volatile.dm
		w_o = gen.w_o.detach().cpu().numpy()
		incoming.result.sent_str = sent_str = \
				[dm.convert_ids_to_sentence(w_o[i].tolist()) for i in range(batch_size)]
		incoming.result.golden_str = golden_str = \
				[incoming.data.sent_str[i] for i in range(batch_size)]
		incoming.result.show_str = "\n".join(["sent: " + a + "\n" + \
				"golden: " + b + "\n" \
				for a, b in zip(sent_str, golden_str)])
