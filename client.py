import models, torch, copy

device = None
"""
如果使用cuda的话，把该段代码改为
if torch.cuda.is_available():
	device = torch.device('cuda')
else device = torch.device('cpu')
"""
if torch.backends.mps.is_available():
	device = torch.device('mps')
else:
	device = torch.device('cpu')

class Client(object):
	def __init__(self, conf, model, train_dataset, id = -1):
		self.conf = conf
		self.local_model = model
		self.client_id = id
		self.train_dataset = train_dataset
		
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset) / self.conf['num_models'])
		train_indices = all_range[id * data_len: (id + 1) * data_len]

		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
									sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
									
	def local_train(self, model, c):

		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	
	
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		
		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				data = data.to(device)
				target = target.to(device)
				
				# if torch.cuda.is_available():
				# 	data = data.cuda()
				# 	target = target.cuda()
			
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
			
				optimizer.step()
			print("Client", c,"Epoch %d done." % e)
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			
		return diff
		