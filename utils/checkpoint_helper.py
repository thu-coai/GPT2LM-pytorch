import os
import json
import shutil
import logging

import torch

def float_format(b, eps=1e-5):

	for i in range(10):
		f = ("%." + str(i) + "g") % b
		rf = float(f)
		if (rf - b) < eps * max(abs(b), abs(rf)):
			break

	for i in range(10):
		e = ("%." + str(i) + "e") % b
		q, w = e.split("e")
		w = str(int(w))
		e = q + 'e' + w
		re = float(e)
		if (re - b) < eps * max(abs(b), abs(re)):
			break

	if len(f) <= len(e):
		return f
	else:
		return e

def name_format(name, *args, eps=1e-5):
	res = [name]
	for i in range(len(args) // 2):
		a = args[i * 2]
		b = args[i * 2 + 1]
		if type(b) is str:
			res.append(a + b)
		elif type(b) is float:
			res.append(a + float_format(b, eps=eps))
		elif type(b) is bool:
			res.append(a + "T" if b else "F")
		else:
			res.append(a + str(b))
	return "_".join(res)

class CheckpointManager:
	def __init__(self, log_name, model_dir, checkpoint_steps=1, checkpoint_max_to_keep=100, best_mode=None):
		self.log_name = log_name
		self.model_dir = model_dir
		self.checkpoint_steps = checkpoint_steps
		self.checkpoint_max_to_keep = checkpoint_max_to_keep
		self.checkpoint_list = []
		self.best_checkpoint = ""
		self.best_mode = best_mode
		self.now_step = 0
		self.reset_best_value()

	def reset_best_value(self):
		if self.best_mode == "max":
			self.best_value = -float("inf")
		elif self.best_mode == "min":
			self.best_value = float("inf")

	#TODO: checkpoint_list is not reliable for multiple processing
	#      refer to https://blog.gocept.com/2013/07/15/reliable-file-updates-with-python/
	def load_checkpoint_list(self):
		try:
			with open("%s/checkpoint_list" % self.model_dir, "r") as checkpoint_list_fp:
				return json.load(checkpoint_list_fp)
		except FileNotFoundError:
			return {}

	def save_checkpoint_list(self, dic):
		with open("%s/checkpoint_list" % self.model_dir, "w") as checkpoint_list_fp:
			json.dump(dic, checkpoint_list_fp)

	def update_best(self, value=None):
		if value and self.best_mode:
			if self.best_mode == "max":
				if value > self.best_value:
					self.best_value = value
					return True
				else:
					return False
			elif self.best_mode == "min":
				if value < self.best_value:
					self.best_value = value
					return True
				else:
					return False
		return False

	def save(self, state, filename, value=None):
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		torch.save(state, "%s/%s.model" % (self.model_dir, filename))

		self.now_step += 1
		if self.now_step % self.checkpoint_steps == 0:
			self.checkpoint_list.append(filename)
			if len(self.checkpoint_list) > self.checkpoint_max_to_keep:
				try:
					os.remove("%s/%s.model" % (self.model_dir, self.checkpoint_list[0]))
				except OSError:
					pass
				self.checkpoint_list.pop(0)
		else:
			if len(self.checkpoint_list) > 1:
				try:
					os.remove("%s/%s.model" % (self.model_dir, self.checkpoint_list[-1]))
				except OSError:
					pass
				self.checkpoint_list.pop()
			self.checkpoint_list.append(filename)

		if self.update_best(value):
			shutil.copyfile("%s/%s.model" % (self.model_dir, filename), \
		 		'%s/%s_best.model' % (self.model_dir, self.log_name))
			self.best_checkpoint = "%s_best" % self.log_name

		cp_dict = self.load_checkpoint_list()
		cp_dict["#last#"] = self.log_name
		cp_dict[self.log_name] = {"list": self.checkpoint_list, "best": self.best_checkpoint}
		self.save_checkpoint_list(cp_dict)

	def restore(self, model_name):
		checkpoint_list = self.load_checkpoint_list()
		if model_name == "last":
			find_name = checkpoint_list["#last#"]
		elif model_name == "best":
			find_name = checkpoint_list["#last#"]

		if model_name[-5:] == "_last":
			find_name = model_name[:-5]
			model_name = "last"
		elif model_name[-5:] == "_best":
			find_name = model_name[:-5]
			model_name = "best"

		if model_name == "last":
			model_name = checkpoint_list[find_name]["list"][-1]
		elif model_name == "best":
			model_name = checkpoint_list[find_name]["best"]

		if os.path.isfile("%s/%s.model" % (self.model_dir, model_name)):
			logging.info("loading checkpoint %s", model_name)
			checkpoint = torch.load("%s/%s.model" % (self.model_dir, model_name), \
					map_location=lambda storage, loc: storage)
		elif os.path.isfile("%s/%s" % (self.model_dir, model_name)):
			logging.info("loading checkpoint %s", model_name)
			checkpoint = torch.load("%s/%s" % (self.model_dir, model_name), \
					map_location=lambda storage, loc: storage)
		elif os.path.isfile("%s" % (model_name)):
			logging.info("loading checkpoint %s", model_name)
			checkpoint = torch.load("%s" % (model_name), \
					map_location=lambda storage, loc: storage)
		else:
			raise ValueError("no checkpoint found at %s" % model_name)

		return checkpoint

	def state_dict(self):
		return {key: value for key, value in self.__dict__.items() if key not in \
					{"log_name", "model_dir", "checkpoint_steps", \
					"checkpoint_max_to_keep", "checkpoint_list", "best_mode"}}

	def load_state_dict(self, state_dict):
		self.__dict__.update(state_dict)
