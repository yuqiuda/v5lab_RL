from collections import deque

class DataRecord:
	def __init__(self):
		self.episodes = 0
		self.episodes_rewards = 0
		self.last_100_reward = deque(maxlen=100)
		self.avg_rewards = []



class OutputPipline():
	def __init__(self, out_dir=None, needToFile=False, needToClient=True):
		self.out_dir = out_dir
		self.needToFile = needToFile
		self.needToClient = needToClient

	def toFile(self, data: list):
		pass

	def toClient(self, data: list):
		print(data)
		pass

	def record(self, data):
		if self.needToFile:
			self.toFile(data)
		if self.needToClient:
			self.toClient(data)
