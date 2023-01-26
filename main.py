import torch
from data_load import get_data_loaders
from net import MlpNet
from train import get_loss_and_opt, train_method


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
    
    train_method(trainloader=trainloader,
                 testloader=testloader, net=net, criterion=criterion,
                 optimizer=optimizer, epochs=EPOCHS, device=device)  


if __name__=='__main__':
    main()