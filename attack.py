import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from models.model import Net
from utils.JSMA import perturbation_single
import torch.utils.data as Data
import torch
from tqdm import tqdm

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
batch_size = 10
adver_nums = 100
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data/mnist', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : x.resize_(28 * 28))])), batch_size=batch_size, shuffle=True)

# 这几个变量主要用于之后的测试以及可视化
adver_example_by_JSMA = torch.zeros((batch_size,1,28,28)).to(device)
adver_target = torch.zeros(batch_size).to(device)
clean_example = torch.zeros((batch_size,1,28,28)).to(device)
clean_target = torch.zeros(batch_size).to(device)


model = Net().to(device)
model.load_state_dict(torch.load('./checkpoints/model.pth', weights_only=False))
model.eval()
theta = 1.0 # 扰动值
gamma = 0.1 # 最多扰动特征数占总特征数量的比例
ys_target= 2 # 对抗性样本的标签
for i,(data, target) in enumerate(test_loader):
    if i >= adver_nums / batch_size :
        break
    if i == 0:
        clean_example = data
    else:
        clean_example = torch.cat((clean_example,data),dim = 0)
        
    cur_adver_example_by_JSMA = torch.zeros_like(data).to(device)

    for j in range(batch_size):
        pert_image = perturbation_single(data[j].resize_(1, 28 * 28).numpy(), ys_target, theta, gamma, model)
        cur_adver_example_by_JSMA[j] = torch.from_numpy(pert_image).to(device)
    
    # 使用对抗样本攻击VGG模型
    pred = model(cur_adver_example_by_JSMA).max(1)[1]
    if i == 0:
        adver_example_by_JSMA = cur_adver_example_by_JSMA
        clean_target = target
        adver_target = pred
    else:
        adver_example_by_JSMA = torch.cat((adver_example_by_JSMA , cur_adver_example_by_JSMA), dim = 0)
        clean_target = torch.cat((clean_target,target),dim = 0)
        adver_target = torch.cat((adver_target,pred),dim = 0)

print (adver_example_by_JSMA.shape) # 100*1*28*28
# print (adver_target) # 100
print (clean_example.shape) # 100*1*28*28
# print (clean_target) # 100

adver_dataset = Data.TensorDataset(adver_example_by_JSMA, clean_target)
loader = Data.DataLoader(dataset=adver_dataset, batch_size=batch_size)
correct_num = torch.tensor(0).to(device)
for (data, target) in loader:
	data = data.to(device)
	target = target.to(device)
	pred = model.forward(data).max(1)[1]
	num = torch.sum(pred==target)
	correct_num = correct_num + num
print ('correct rate is {}'.format(correct_num / adver_nums))

def plot_clean_and_adver(adver_example,adver_target,clean_example,clean_target):
	n_cols = 5
	n_rows = 5
	cnt = 1
	cnt1 = 1
	plt.figure(figsize=(n_cols*4,n_rows*2))
	for i in range(n_cols):
		for j in range(n_rows):
			plt.subplot(n_cols,n_rows*2,cnt1)
			plt.xticks([])
			plt.yticks([])
			plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
			plt.imshow(clean_example[cnt].reshape(28,28).to('cpu').detach().numpy(),cmap='gray')
			plt.subplot(n_cols,n_rows*2,cnt1+1)
			plt.xticks([])
			plt.yticks([])
			plt.imshow(adver_example[cnt].reshape(28,28).to('cpu').detach().numpy(),cmap='gray')
			cnt = cnt + 1
			cnt1 = cnt1 + 2
	plt.show()

plot_clean_and_adver(adver_example_by_JSMA,adver_target,clean_example,clean_target)
