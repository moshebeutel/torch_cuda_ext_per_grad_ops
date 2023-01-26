import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from per_sample_grad import get_per_sample_grads


LOG = {}


def get_loss_and_opt(net, learning_rate):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    return criterion, optimizer


def train_single_epoch(trainloader, net, criterion, optimizer, device, grads_manipulation = None):
    net.train()

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Transfer Data to Device
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_size = labels.shape[0]

        optimizer.zero_grad()
        
        if(grads_manipulation is None):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
        else:
            per_sample_grads = get_per_sample_grads(net=net, criterion=criterion, labels=labels, inputs=inputs)
            for ii,p in enumerate(net.parameters()):
                g = per_sample_grads[ii].detach()
                p.grad = grads_manipulation(g)
        

        ## sign sgd
        # for p in net.parameters():
        #     p.grad = torch.sign(p.grad)

        optimizer.step()


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

        train_single_epoch(trainloader=trainloader, net=net, criterion=criterion, optimizer=optimizer, device=device, grads_manipulation=lambda g: g.mean(dim=0))  
        eval_net(testloader=testloader, net=net, device=device)

        val_acc = LOG['val_acc']

        prog_bar.set_description(f'Epoch {epoch}: Val Acc {val_acc}')

