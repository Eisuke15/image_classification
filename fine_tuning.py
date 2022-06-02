import torch
import torch.utils.data as data
from torch import nn, optim
from torchvision.models import vgg16
from tqdm import tqdm

from transfar_learning import device, train_dataset, val_dataset

batch_size = 32

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


net = vgg16(pretrained=True)
net.classifier[6] = nn.Linear(net.classifier[6].in_features, 1)

net.train()

criterion = nn.BCEWithLogitsLoss()


params1 = net.features.parameters()
params2 = list(net.classifier[0].parameters()) + list(net.classifier[3].parameters())
params3 = net.classifier[6].parameters()


optimizer = optim.SGD([
    {'params': params1, 'lr': 1e-4},
    {'params': params2, 'lr': 5e-4},
    {'params': params3, 'lr': 1e-3},
], momentum=0.9)


net.to(device)

torch.backends.cudnn.benchmark = True

num_epochs = 10


for epoch in range(num_epochs):
    net.train()

    train_loss = 0.0
    train_corrects = 0
    train_num = 0

    if (epoch != 0): # no train at epoch 0 to check the performance with no train.
        for inputs, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            train_num += inputs.size(0)
            outputs = net(inputs).squeeze()
            loss = criterion(outputs, labels.float())

            preds = torch.where(outputs > 0, 1, 0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels)

    net.eval()

    eval_loss = 0.0
    eval_corrects = 0
    eval_num = 0

    for inputs, labels in tqdm(val_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        eval_num += inputs.size(0)
        outputs = net(inputs).squeeze()
        loss = criterion(outputs, labels.float())

        preds = torch.where(outputs > 0, 1, 0)

        eval_loss += loss.item() * inputs.size(0)
        eval_corrects += torch.sum(preds == labels)

    if epoch:
        print(f"epoch: {epoch + 1}  train loss: {train_loss/train_num}  train acc: {train_corrects.double()/train_num}  eval loss: {eval_loss/eval_num}  eval acc: {eval_corrects.double()/eval_num}")

    else:
        print(f"epoch: {epoch + 1} eval loss: {eval_loss/eval_num}  eval acc: {eval_corrects.double()/eval_num}")


