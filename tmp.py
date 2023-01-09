import torch
from  torchvision.models import resnet18
from pprint import pprint


model=resnet18(pretrained=True)
model=model.to('cuda')
inputs= torch.randn(1,3,224,224).to('cuda')
model.train()
output=model(inputs)
snapshot=torch.cuda.memory_snapshot()
pprint(snapshot)



from pickle import dump
dump(snapshot,open("snapshot.pickle","wb"))

