import torch 
import os
from tqdm import tqdm
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from model import Generator, Discriminator
from utils import D_train, G_train, save_models, Divergence

# changes: can use cpu for local debugging,
#   added version argument for tensorboard,
#   added writer for tensorboard


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD.")
    parser.add_argument("--divergence", "-div", type=str,
                        choices=["gan",
                                 "pr",
                                 'total_variation',
                                 'forward_kl',
                                 'reverse_kl',
                                 'pearson',
                                 'hellinger',
                                 'jensen_shannon'])
    parser.add_argument("--version", "-v", type=str, default="test",
                        help="Name of run for Tensorboard.")

    args = parser.parse_args()


    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    #transform = transforms.Compose([
    #            transforms.ToTensor(),
    #            transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transforms.ToTensor(), download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim=mnist_dim)).to(device)
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).to(device)

    print('Model loaded.')

    # define writer to accompany training
    writer = SummaryWriter(log_dir=f"tb_logs/{args.version}")

    # define loss
    criterion = Divergence(args.divergence)

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in range(n_epoch):
        with tqdm(enumerate(train_loader), total=len(train_loader),
                  leave=False, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (x, _) in pbar:
                log = (writer, batch_idx, epoch, len(train_loader))

                x = x.view(-1, mnist_dim)
                D_train(x, G, D, D_optimizer, criterion, device, log=log)
                G_train(x, G, D, G_optimizer, criterion, device, log=log)

            if (epoch + 1) % 10 == 0:
                save_models(G, D, 'checkpoints')
                
    print('Training done')

    writer.close()
        