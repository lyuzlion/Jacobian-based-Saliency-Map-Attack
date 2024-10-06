import torch
from torchvision import datasets, transforms
import torch.nn as nn
from tqdm import tqdm
from models.model import Net

test_data = datasets.MNIST('../data/mnist', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : x.resize_(28 * 28))]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True)

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load('./checkpoints/model.pth', weights_only=False))
model.eval()

correct_num = torch.tensor(0).to(device)

for j, (data,target) in tqdm(enumerate(test_loader)):
    data = data.to(device)
    target = target.to(device)
    logit = model(data)
    pred = logit.max(1)[1]
    num = torch.sum(pred==target)
    correct_num = correct_num + num
print ('correct rate is {}'.format(correct_num/10000))
