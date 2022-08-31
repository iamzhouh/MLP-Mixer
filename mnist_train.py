from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from MLPMixerModel import *


batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差

train_dataset = datasets.MNIST(root='./dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='./dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MLPMixer(in_channels=1, dim=64, num_classes=10, patch_size=4, image_size=28, depth=2, token_dim=128, channel_dim=128)
#                in_channels=1, dim=64, num_classes=10, patch_size=4, image_size=28, depth=1, token_dim=64, channel_dim=128    95%
#                in_channels=1, dim=64, num_classes=10, patch_size=4, image_size=28, depth=2, token_dim=64, channel_dim=128    97%
#                in_channels=1, dim=64, num_classes=10, patch_size=4, image_size=28, depth=3, token_dim=64, channel_dim=128    97%
#                in_channels=1, dim=64, num_classes=10, patch_size=4, image_size=28, depth=2, token_dim=128, channel_dim=512   97%
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader, 0):
        inputs, labels = data
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

    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)
