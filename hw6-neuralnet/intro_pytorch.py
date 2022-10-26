import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_data_loader(training = True):
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=custom_transform)
    test_set = datasets.FashionMNIST('./data', train=False, transform=custom_transform)

    loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    if not training:
        loader = torch.utils.data.DataLoader(test_set, batch_size=64)
    
    return loader

def build_model():
    # a flatten layer to convert 2D array to 1D array
    # a dense layer with 128 nodes and a ReLU activation
    # a dense layer with 64 nodes and a ReLU activation
    # a dense layer with 10 nodes

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    return model

def train_model(model, train_loader, criterion, T):
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
   
    for epoch in range(T):
        running_loss = 0.0
        num_correct = 0
        total = 0

        model.train()
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
                
            running_loss += loss.item() * 64
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            num_correct += (predicted == labels).sum().item()

        acc = 100 * num_correct / total
        loss = running_loss / total
        print(f'Train Epoch: {epoch} Accuracy: {num_correct}/{total}({acc:.2f}%) Loss: {loss:.3f}')

def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()

    running_loss = 0.0
    num_correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * 64
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            num_correct += (predicted == labels).sum().item()

    loss = running_loss / total
    acc = 100 * num_correct / total

    if show_loss:
        print(f'Average loss: {loss:.4f}')
    
    print(f'Accuracy: {acc:.2f}%')

def predict_label(model, test_images, index):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    logits = model(test_images)
    probs = F.softmax(logits, dim=1)
    
    top3_prob = torch.topk(probs[index], 3)[0]
    top3_prob = top3_prob.detach().numpy()

    top3_idx = torch.topk(probs[index], 3)
    top3_idx = top3_idx.indices.detach().numpy()

    for i in range(3): 
        print(f'{class_names[top3_idx[i]]}: {top3_prob[i] * 100:.2f}%')


if __name__ == '__main__':
    # train_loader = get_data_loader()
    # test_loader = get_data_loader(training=False)
    
    # model = build_model()
    
    criterion = nn.CrossEntropyLoss()
    
    # train_model(model, train_loader, criterion, 5)
    
    # evaluate_model(model, test_loader, criterion, show_loss = True)
    
    # pred_set, _ = next(iter(test_loader))
    # predict_label(model, pred_set, 1)
