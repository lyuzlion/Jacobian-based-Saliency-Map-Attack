from torchvision import datasets, transforms
import torch
from models.model import Net
from tqdm import tqdm

mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : x.resize_(28 * 28))])
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=False, download=False, transform=mnist_transform),
        batch_size=10, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=False, transform=mnist_transform),
        batch_size=10, shuffle=True)

batch_size = 10
epoch = 10
learning_rate = 0.001
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss().to(device)
max_correct = 0

for i in tqdm(range(1, epoch + 1)):
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        logit = model(data)
        loss = criterion(logit, target)
        model.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        correct_num = torch.tensor(0).to(device)
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            logit = model(data)
            pred = logit.max(1)[1]
            num = torch.sum(pred == target)
            correct_num = correct_num + num
        print('\n{} correct rate is {}'.format(i, correct_num / 10000))
        if correct_num > max_correct:
            max_correct = correct_num
            torch.save(model.state_dict(), './checkpoints/model.pth')
            print('model saved')

