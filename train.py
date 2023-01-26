import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

LOG = {}


def get_loss_and_opt(net, learning_rate):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    return criterion, optimizer


def train_single_epoch(trainloader, net, criterion, optimizer, device):
    net.train()
    correct = 0
    total = 0

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        # Transfer Data to Device
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_size = labels.shape[0]

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()


        ## sign sgd
        # for p in net.parameters():
        #     p.grad = torch.sign(p.grad)

        
        max_probs, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.step()

    LOG['train_acc'] = 100 * correct / total


def eval_net(testloader, net, device):
    net.eval()
    correct = 0
    total = 0

    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Calculate outputs by running images through the network
            outputs = net(inputs)
            
            # The class with the highest probability is what we choose as prediction
            max_probs, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            


    LOG['val_acc'] = 100 * correct / total



def train_method(trainloader, testloader, net, criterion, optimizer, epochs=50, device='cpu'):

    prog_bar = tqdm(range(epochs))
    for epoch in prog_bar:

        train_single_epoch(trainloader=trainloader, net=net, criterion=criterion, optimizer=optimizer, device=device)  
        eval_net(testloader=testloader, net=net, device=device)

        train_acc = LOG['train_acc']
        val_acc = LOG['val_acc']

        prog_bar.set_description(f'Epoch {epoch}: Train Acc {train_acc} Val Acc {val_acc}')

