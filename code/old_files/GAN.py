import math
import time
from pathlib import Path
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_functions.twitter_dataset import TwitterDataset
from dataset_functions.graph_dataset import GraphDataset
from classes.basic_classes import DataSet

##########################
### SETTINGS
##########################
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Hyperparameters
# Remember that GANs are highly sensitive to hyper-parameters
random_seed = 123
generator_learning_rate = 0.001
discriminator_learning_rate = 0.001
BATCH_SIZE = 128
LATENT_DIM = 100  # latent vectors dimension [z]
IMG_SHAPE = (1, 28, 28)  # MNIST has 1 color channel, each image 28x8 pixels
IMG_SIZE = 1

# for x in IMG_SHAPE:
#     IMG_SIZE *= x

# train_dataset = datasets.MNIST(root='./datasets',
#                                train=True,
#                                transform=transforms.ToTensor(),
#                                download=True)
# test_dataset = datasets.MNIST(root='./datasets',
#                               train=False,
#                               transform=transforms.ToTensor())
# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=BATCH_SIZE,
#                           shuffle=True)
# test_loader = DataLoader(dataset=test_dataset,
#                          batch_size=BATCH_SIZE,
#                          shuffle=False)
# constant the seed
torch.manual_seed(random_seed)


# build the model, send it ti the device
# model = GAN().to(device)
# optimizers: we have one for the generator and one for the discriminator
# that way, we can update only one of the modules, while the other one is "frozen"
# optim_gener = torch.optim.Adam(model.generator.parameters(), lr=generator_learning_rate)
# optim_discr = torch.optim.Adam(model.discriminator.parameters(), lr=discriminator_learning_rate)


def something():
    raise NotImplementedError


# class GAN(torch.nn.Module):
#     def __init__(self):
#         super(GAN, self).__init__()
#
#         # generator: z [vector] -> image [matrix]
#         self.generator = nn.Sequential(
#             nn.Linear(LATENT_DIM, 128),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(128, IMG_SIZE),
#             nn.Tanh()
#         )
#
#         # discriminator: image [matrix] -> label (0-fake, 1-real)
#         self.discriminator = nn.Sequential(
#             nn.Linear(IMG_SIZE, 128),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
#
#     def generator_forward(self, z):
#         img = self.generator(z)
#         return img
#
#     def discriminator_forward(self, img):
#         pred = model.discriminator(img)
#         return pred.view(-1)
#
#     def discriminator_forward(self, img):
#         pred = model.discriminator(img)
#         return pred.view(-1)
#         start_time = time.time()


class GANTrainer:
    def __init__(self, att_model, att_optimzer, def_model, def_optimzer,
                 att_loss_fn=binary_cross_entropy,
                 def_loss_fn=binary_cross_entropy,
                 dataset=GraphDataset(DataSet.TWITTER, device),
                 lam=0.5,
                 patience=math.inf
                 ):
        self.att_model = att_model
        self.att_optimzer = att_optimzer
        self.def_model = def_model
        self.def_optimzer = def_optimzer
        self.dataset = dataset
        self.att_loss_fn = att_loss_fn
        self.def_loss_fn = def_loss_fn
        self.lam = lam
        self.patience = patience

    def train(self, num_of_epochs):

        #########################
        # Define checkpoint files
        #########################
        checkpoint_def = "def_model_weights"
        checkpoint_att = "att_model_weights"
        checkpoint_def_filename = f"{checkpoint_def}{str(time.time())}"
        print(checkpoint_def)
        Path(os.path.dirname(checkpoint_def_filename)).mkdir(exist_ok=True)
        checkpoint_att_filename = f"{checkpoint_att}{str(time.time())}"
        print(checkpoint_att)
        Path(os.path.dirname(checkpoint_att_filename)).mkdir(exist_ok=True)

        ################################
        # Define early stopping vraibles
        ################################
        patience_counter, best_val_accuracy = 0, 0

        self.start_time = time.time()
        self.discr_costs = []
        self.gener_costs = []

        best_att_val_filename = None
        best_def_val_filename = None
        for epoch in range(num_of_epochs):
            # model = model.train()
            # for batch_idx, (features, targets) in enumerate(train_loader):
            #     features = (features - 0.5) * 2.0  # normalize between [-1, 1]
            # features = features.view(-1, IMG_SIZE).to(device)
            # targets = targets.to(device)
            # generate fake and real labels
            # valid = torch.ones(targets.size(0)).float().to(device)
            # fake = torch.zeros(targets.size(0)).float().to(device)

            gener_loss, discr_loss = self.foreach_epoch()

            self.discr_costs.append(discr_loss)
            self.gener_costs.append(gener_loss)
            # if not batch_idx % 100:
            print(f'Epoch: {epoch} | Attacker Loss: {gener_loss} | Defender loss: {discr_loss}')

            print('Time elapsed: %.2f min' % ((time.time() - self.start_time) / 60))

            print('Total Training Time: %.2f min' % ((time.time() - self.start_time) / 60))

            # save weigths and preform test pass
            # Saving attack weights
            torch.save(self.att_model.state_dict(), checkpoint_att_filename + f"_{epoch}.pt")
            print(
                f"*** Saved attack checkpoint {checkpoint_att_filename}_{epoch}.pt at epoch {epoch + 1}"
            )
            # Saving defence weights
            torch.save(self.def_model.state_dict(), checkpoint_def_filename + f"_{epoch}.pt")
            print(
                f"*** Saved defense checkpoint {checkpoint_def_filename}_{epoch}.pt at epoch {epoch + 1}"
            )

            att_loss, real_loss, fake_loss, def_loss = self.test()

            val_acc = def_loss
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_att_val_filename = f"{checkpoint_att_filename}_{epoch}.pt"
                best_def_val_filename = f"{checkpoint_def_filename}_{epoch}.pt"
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                break

        print(f"best accuracy weights are saved in {best_att_val_filename} and {best_def_val_filename}")

    def foreach_epoch(self):
        ### FORWARD PASS AND BACKPROPAGATION

        # --------------------------
        # Train Attacker
        # --------------------------

        # Freezing defender weights
        # SINGLE attacking the current network

        # TODO do we pass over the whole dataset as the network input?
        attack_output = self.att_model.forward(self.dataset.data)

        # creating tagging for loss
        fake_valid = something(self.dataset.data, attack_output)
        # TODO: create the valid network with valid tagging on the attack node
        # and malicious tagging on the attacked node, all other are same as real tagging

        # Loss for fooling the GAL
        # TODO: think if running with no_grad()?
        # TODO: Create network with attack node and no tagging
        def_pred = self.def_model.forward(attack_output)

        # here we use the `valid` labels because we want the defender to "think"
        # the attacked samples are real
        att_loss = self.att_loss_fn(def_pred, fake_valid)

        self.att_optimzer.zero_grad()
        att_loss.backward()
        self.att_optimzer.step()

        # --------------------------
        # Train Defender
        # --------------------------

        # Freezing Attacker weights
        # TODO get labels from data
        real_tagging = something(self.dataset.data)
        # Real forward pass
        def_tagging_real = self.def_model.forward(self.dataset.data)
        real_loss = self.def_loss_fn(def_tagging_real, real_tagging)

        # TODO generate the currect tagging for the network with attack node
        currect_tagging_attacked = something(real_tagging, attack_output)
        # here we use the `fake` labels when training the discriminator
        # TODO: Create network with attack node and no tagging
        def_tagging_attacked = self.def_model.forward(attack_output)
        fake_loss = self.def_loss_fn(def_tagging_attacked, currect_tagging_attacked)

        def_loss = ((1 - self.lam) * real_loss + self.lam * fake_loss)
        self.def_optimzer.zero_grad()
        def_loss.backward()
        self.def_optimzer.step()

        return att_loss, def_loss

    def test(self):
        with torch.no_grad():
            # TODO do we pass over the whole dataset as the network input?
            attack_output = self.att_model.forward(self.dataset.data.test)  # TODO - get test dataset
            fake_valid = something(self.dataset.data.test, attack_output)  # TODO - get test dataset
            # TODO: create the valid network with valid tagging on the attack node
            # TODO: think if running with no_grad()?
            # TODO: Create network with attack node and no tagging
            def_pred = self.def_model.forward(attack_output)
            att_loss = self.att_loss_fn(def_pred, fake_valid)

            # TODO get labels from data
            real_tagging = something(self.dataset.data.test)  # TODO - get test dataset
            def_tagging_real = self.def_model.forward(self.dataset.data.test)  # TODO - get test dataset
            real_loss = self.def_loss_fn(def_tagging_real, real_tagging)

            # TODO generate the currect tagging for the network with attack node
            currect_tagging_attacked = something(real_tagging, attack_output)
            fake_loss = self.def_loss_fn(def_pred, currect_tagging_attacked)
            def_loss = ((1 - self.lam) * real_loss + self.lam * fake_loss)

            return att_loss, real_loss, fake_loss, def_loss

        return

    def evaluate(self, num_of_epochs):

        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(range(len(self.gener_costs)), self.gener_costs, label='Attacker loss')
        ax1.plot(range(len(self.discr_costs)), self.discr_costs, label='Defender loss')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.legend()
        # Set scond x-axis
        ax2 = ax1.twiny()
        newlabel = list(range(num_of_epochs + 1))
        iter_per_epoch = len(train_loader)
        newpos = [e * iter_per_epoch for e in newlabel]
        ax2.set_xticklabels(newlabel[::10])
        ax2.set_xticks(newpos[::10])
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 45))
        ax2.set_xlabel('Epochs')
        ax2.set_xlim(ax1.get_xlim())


def main():
    pass


if __name__ != '__main__':
    main()
