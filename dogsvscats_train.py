from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from MLPMixerModel import *
from dogsvscats_datapreprocess import *

batch_size = 16
# datasets_dir = "/newdisk/zh/dogs-vs-cats"
datasets_dir = "D:/dataset/dogs-vs-cats"

transform = transforms.Compose([
    transforms.ToTensor()
])  # 归一化,均值和方差

train_dataset = DogsVSCatsDataset('train', datasets_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = DogsVSCatsDataset('test', datasets_dir)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MLPMixer(in_channels=3, dim=512, num_classes=2, patch_size=8, image_size=200, depth=3, token_dim=512, channel_dim=1024)
#                in_channels=3, dim=512, num_classes=2, patch_size=8, image_size=200, depth=2, token_dim=512, channel_dim=1024    epoch=10  97%
#                in_channels=3, dim=512, num_classes=2, patch_size=8, image_size=200, depth=3, token_dim=512, channel_dim=1024    epoch=20  99%
model.to(device)

criterion = torch.nn.CrossEntropyLoss()     # torch.nn.CrossEntropyLoss自带了softmax
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = labels.reshape(-1)

        inputs, labels = inputs.to(device), labels.to(device)

        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index%300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            labels = labels.reshape(-1)

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))
    return correct / total

if __name__ == "__main__":
    epoch_list = []
    acc_list = []
    print("使用" + str(device) + "训练")

    for epoch in range(20):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)


# ssh://root@10.3.125.120:22/home/anaconda3/envs/zhouh_pytorch/bin/python3.7 -u /newdisk/zh/MLP-Mixer/dogsvscats_train.py
# 使用cuda:0训练
# [1,   300] loss: 0.705
# [1,   600] loss: 0.666
# [1,   900] loss: 0.650
# [1,  1200] loss: 0.625
# accuracy on test set: 65 %
# [2,   300] loss: 0.618
# [2,   600] loss: 0.613
# [2,   900] loss: 0.597
# [2,  1200] loss: 0.593
# accuracy on test set: 68 %
# [3,   300] loss: 0.576
# [3,   600] loss: 0.578
# [3,   900] loss: 0.574
# [3,  1200] loss: 0.576
# accuracy on test set: 71 %
# [4,   300] loss: 0.553
# [4,   600] loss: 0.552
# [4,   900] loss: 0.554
# [4,  1200] loss: 0.545
# accuracy on test set: 73 %
# [5,   300] loss: 0.515
# [5,   600] loss: 0.520
# [5,   900] loss: 0.522
# [5,  1200] loss: 0.508
# accuracy on test set: 79 %
# [6,   300] loss: 0.468
# [6,   600] loss: 0.489
# [6,   900] loss: 0.467
# [6,  1200] loss: 0.487
# accuracy on test set: 83 %
# [7,   300] loss: 0.410
# [7,   600] loss: 0.407
# [7,   900] loss: 0.431
# [7,  1200] loss: 0.415
# accuracy on test set: 87 %
# [8,   300] loss: 0.290
# [8,   600] loss: 0.322
# [8,   900] loss: 0.341
# [8,  1200] loss: 0.355
# accuracy on test set: 93 %
# [9,   300] loss: 0.160
# [9,   600] loss: 0.215
# [9,   900] loss: 0.227
# [9,  1200] loss: 0.262
# accuracy on test set: 97 %
# [10,   300] loss: 0.099
# [10,   600] loss: 0.125
# [10,   900] loss: 0.155
# [10,  1200] loss: 0.168
# accuracy on test set: 97 %
# [11,   300] loss: 0.068
# [11,   600] loss: 0.095
# [11,   900] loss: 0.109
# [11,  1200] loss: 0.127
# accuracy on test set: 96 %
# [12,   300] loss: 0.052
# [12,   600] loss: 0.062
# [12,   900] loss: 0.088
# [12,  1200] loss: 0.096
# accuracy on test set: 98 %
# [13,   300] loss: 0.037
# [13,   600] loss: 0.052
# [13,   900] loss: 0.048
# [13,  1200] loss: 0.077
# accuracy on test set: 99 %
# [14,   300] loss: 0.031
# [14,   600] loss: 0.036
# [14,   900] loss: 0.060
# [14,  1200] loss: 0.060
# accuracy on test set: 98 %
# [15,   300] loss: 0.032
# [15,   600] loss: 0.036
# [15,   900] loss: 0.036
# [15,  1200] loss: 0.039
# accuracy on test set: 99 %
# [16,   300] loss: 0.025
# [16,   600] loss: 0.023
# [16,   900] loss: 0.040
# [16,  1200] loss: 0.049
# accuracy on test set: 99 %
# [17,   300] loss: 0.015
# [17,   600] loss: 0.027
# [17,   900] loss: 0.027
# [17,  1200] loss: 0.027
# accuracy on test set: 99 %
# [18,   300] loss: 0.011
# [18,   600] loss: 0.013
# [18,   900] loss: 0.025
# [18,  1200] loss: 0.038
# accuracy on test set: 99 %
# [19,   300] loss: 0.023
# [19,   600] loss: 0.014
# [19,   900] loss: 0.024
# [19,  1200] loss: 0.029
# accuracy on test set: 99 %
# [20,   300] loss: 0.009
# [20,   600] loss: 0.012
# [20,   900] loss: 0.011
# [20,  1200] loss: 0.024
# accuracy on test set: 99 %
#
# Process finished with exit code 0