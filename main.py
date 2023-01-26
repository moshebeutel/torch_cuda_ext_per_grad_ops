import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from data_load import get_data_loaders
from net import MlpNet

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



def main():

    BATCH_SIZE = 128
    EPOCHS = 1000
    LEARNING_RATE = 0.0001
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Device:', device)
    
    trainloader, testloader = get_data_loaders(batch_size=BATCH_SIZE)
    print('train loader', len(trainloader))

    # Transfer NN parameters to device - done in Pytorch
    net = MlpNet().to(device)  

    criterion, optimizer = get_loss_and_opt(net, learning_rate=LEARNING_RATE)
 
    # privacy_engine = PrivacyEngine()
    # priv_model, priv_optimizer, priv_loader = privacy_engine.make_private(
    #     module=net,
    #     optimizer=optimizer,
    #     data_loader=trainloader,
    #     noise_multiplier=1.1,
    #     max_grad_norm=1.0,
    # )

    # train_method(trainloader=priv_loader, testloader=testloader, net=priv_model, criterion=criterion,
    #              optimizer=priv_optimizer, epochs=EPOCHS)  

    
    train_method(trainloader=trainloader, testloader=testloader, net=net, criterion=criterion,
                 optimizer=optimizer, epochs=EPOCHS, device=device)  


if __name__=='__main__':
    main()