import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# changes: added tensorboard support

class Divergence:
    """
    Taken from https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py#L85
    We added "gan" and "pr" modes.
    """
    def __init__(self, method):
        self.method = method.lower().strip()
        assert self.method in ["gan",
                               "pr",
                               'total_variation',
                               'forward_kl',
                               'reverse_kl',
                               'pearson',
                               'hellinger',
                               'jensen_shannon'], \
            'Invalid divergence.'

        self.pr_control = 10

    def D_loss(self, DX_score, DG_score):
        """ Compute batch loss for discriminator using f-divergence metric """

        if self.method == 'total_variation':
            return -(torch.mean(0.5*torch.tanh(DX_score)) \
                        - torch.mean(0.5*torch.tanh(DG_score)))

        elif self.method == 'forward_kl':
            return -(torch.mean(DX_score) - torch.mean(torch.exp(DG_score-1)))

        elif self.method == 'reverse_kl':
            return -(torch.mean(-torch.exp(DX_score)) - torch.mean(-1-DG_score))

        elif self.method == 'pearson':
            return -(torch.mean(DX_score) - torch.mean(0.25*DG_score**2 + DG_score))

        elif self.method == 'hellinger':
            return -(torch.mean(1-torch.exp(DX_score)) \
                        - torch.mean((1-torch.exp(DG_score))/(torch.exp(DG_score))))

        elif self.method == 'jensen_shannon':
            return -(torch.mean(torch.tensor(2.)-(1+torch.exp(-DX_score))) \
                        - torch.mean(-(torch.tensor(2.)-torch.exp(DG_score))))

        elif self.method == "gan":
            return -(torch.mean(-torch.log(1 + torch.exp(-DX_score))) \
                     - torch.mean(torch.log(1 + torch.exp(DG_score))))

        elif self.method == "pr":
            acc = 0 if self.pr_control <= 1 else self.pr_control - 1
            return -(torch.mean(self.pr_control * torch.sigmoid(DX_score))
                     - torch.mean(torch.sigmoid(DG_score) + acc))

    def G_loss(self, DG_score):
        """ Compute batch loss for generator using f-divergence metric """

        if self.method == 'total_variation':
            return -torch.mean(0.5*torch.tanh(DG_score))

        elif self.method == 'forward_kl':
            return -torch.mean(torch.exp(DG_score-1))

        elif self.method == 'reverse_kl':
            return -torch.mean(-1-DG_score)

        elif self.method == 'pearson':
            return -torch.mean(0.25*DG_score**2 + DG_score)

        elif self.method == 'hellinger':
            return -torch.mean((1-torch.exp(DG_score))/(torch.exp(DG_score)))

        elif self.method == 'jensen_shannon':
            return -torch.mean(-(torch.tensor(2.)-torch.exp(DG_score)))

        elif self.method == "gan":
            return -torch.mean(torch.log(1 + torch.exp(DG_score)))

        elif self.method == "pr":
            return torch.mean(((torch.sigmoid(DG_score) * (1 - torch.sigmoid(DG_score)))
                               - 1) ** 2)


def D_train(x, G, D, D_optimizer, criterion, device, log=None):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output_real = D(x_real)
    #D_real_loss = criterion(D_output, y_real)
    #D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)

    D_output_fake = D(x_fake)
    
    #D_fake_loss = criterion(D_output, y_fake)
    #D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = criterion.D_loss(D_output_real, D_output_fake)
    D_loss.backward()

    #nn.utils.clip_grad_norm_(D.parameters(), 1e-3)

    D_optimizer.step()

    # log to tensorboard
    if log is not None:
        discriminator_log(log, D_loss, D_loss, D_loss)
        
    return D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, device, log=None):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)
                 
    G_output = G(z)
    D_output = D(G_output)
    #G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss = criterion.G_loss(D_output)
    G_loss.backward()

    #nn.utils.clip_grad_norm_(G.parameters(), 1e-3)

    G_optimizer.step()

    # log to tensorboard
    if log is not None:
        generator_log(log, x, G_output, G_loss)
        
    return G_loss.data.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G



def generator_log(log, real_batch, fake_batch, loss):
    writer, idx, ep, num_batches = log
    writer.add_scalar("G_loss/train_step", loss.item(),
                      global_step=ep * num_batches + idx)

    if idx % 200 == 0:
        for i in range(0, real_batch.shape[0], 8):
            img = fake_batch[i].reshape(28, 28).cpu().detach().numpy()
            fig, ax = plt.subplots(1, 1)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            fig.tight_layout()
            writer.add_figure("samples/{:04}".format(i), fig,
                              global_step=ep * num_batches + idx)
            plt.close(fig)

        ref = real_batch[0].reshape(28, 28).cpu().detach().numpy()
        fig, ax = plt.subplots(1, 1)
        ax.imshow(ref, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fig.tight_layout()
        writer.add_figure("reference", fig,
                          global_step=ep * num_batches + idx)
        plt.close(fig)

def discriminator_log(log, real_loss, fake_loss, loss):
    writer, idx, ep, num_batches = log
    writer.add_scalar("D_loss/train_step", loss.item(),
                      global_step=ep * num_batches + idx)
    writer.add_scalar("D_loss/real_train_step", real_loss.item(),
                      global_step=ep * num_batches + idx)
    writer.add_scalar("D_loss/fake_train_step", fake_loss.item(),
                      global_step=ep * num_batches + idx)