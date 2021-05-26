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
from datasets.twitter_dataset import TweeterDataset
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
# NUM_EPOCHS = 100
BATCH_SIZE = 128
LATENT_DIM = 100 # latent vectors dimension [z]
IMG_SHAPE = (1, 28, 28) # MNIST has 1 color channel, each image 28x8 pixels
IMG_SIZE = 1

for x in IMG_SHAPE:
    IMG_SIZE *= x

train_dataset = datasets.MNIST(root='./datasets',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = datasets.MNIST(root='./datasets',
                           train=False,
                           transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset,
                       batch_size=BATCH_SIZE,
                       shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                      batch_size=BATCH_SIZE,
                      shuffle=False)
# constant the seed
torch.manual_seed(random_seed)
# build the model, send it ti the device
model = GAN().to(device)
# optimizers: we have one for the generator and one for the discriminator
# that way, we can update only one of the modules, while the other one is "frozen"
optim_gener = torch.optim.Adam(model.generator.parameters(), lr=generator_learning_rate)
optim_discr = torch.optim.Adam(model.discriminator.parameters(), lr=discriminator_learning_rate)


def something():
    raise NotImplementedError

class GAN(torch.nn.Module):
    def __init__(self):
        super(GAN, self).__init__()

        # generator: z [vector] -> image [matrix]
        self.generator = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, IMG_SIZE),
            nn.Tanh()
        )

        # discriminator: image [matrix] -> label (0-fake, 1-real)
        self.discriminator = nn.Sequential(
        nn.Linear(IMG_SIZE, 128),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(128, 1),
        nn.Sigmoid()
        )

    def generator_forward(self, z):
         img = self.generator(z)
         return img

    def discriminator_forward(self, img):
        pred = model.discriminator(img)
        return pred.view(-1)

    def discriminator_forward(self, img):
        pred = model.discriminator(img)
        return pred.view(-1)
        start_time = time.time()


class GANTrainer:

    def __init__(self, att_model, att_optimzer, def_model, def_optimzer,
                 att_loss_fn=binary_cross_entropy,
                 def_loss_fn=binary_cross_entropy,
                 dataset=TweeterDataset('data/'),
                 lam=0.5,
                 patience=math.inf
                ):
        self.att_model     = att_model
        self.att_optimzer  = att_optimzer
        self.def_model     = def_model
        self.def_optimzer  = def_optimzer
        self.dataset       = dataset
        self.att_loss_fn   = att_loss_fn
        self.def_loss_fn   = def_loss_fn
        self.lam           = lam
        self.patience      = patience

    def train(self, epochs):

        #########################
        # Define checkpoint files
        #########################
        checkpoint_def = "def_model_weights"
        checkpoint_att = "att_model_weights"
        checkpoint_def_filename = f"{checkpoint_def}{str(time.time())}.pt"
        print(checkpoint_def)
        Path(os.path.dirname(checkpoint_def_filename)).mkdir(exist_ok=True)
        checkpoint_att_filename = f"{checkpoint_att}{str(time.time())}.pt"
        print(checkpoint_att)
        Path(os.path.dirname(checkpoint_att_filename)).mkdir(exist_ok=True)

        ################################
        # Define early stopping vraibles
        ################################
        patience_counter, best_val_accuracy = 0, 0


        self.start_time = time.time()
        self.discr_costs = []
        self.gener_costs = []

        for epoch in range(epochs):
            model = model.train()
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
            torch.save(self.att_model.state_dict(), checkpoint_att_filename)
            print(
                f"*** Saved checkpoint {checkpoint_att_filename} " f"at epoch {epoch + 1}"
            )
                # Saving defence weights
            torch.save(self.def_model.state_dict(), checkpoint_def_filename)
            print(
                f"*** Saved checkpoint {checkpoint_def_filename} " f"at epoch {epoch + 1}"
            )

            test_result = self.test()

            val_acc = test_result[0]
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                break


    def foreach_epoch(self):
        ### FORWARD PASS AND BACKPROPAGATION

        # --------------------------
        # Train Attacker
        # --------------------------

        # Freezing defender weights
        # SINGLE attacking the current network

        #TODO do we pass over the whole dataset as the network input?
        attack_output = self.att_model.forward(self.dataset.data)

        # creating tagging for loss
        fake_valid = something(self.dataset.data, attack_output)
        # TODO: create the valid network with valid tagging on the attack node
        # and malicious tagging on the attacked node, all other are same as real tagging

        # Loss for fooling the GAL
        # TODO: think if running with no_grad()?
        # TODO: Create network with attack node and no tagging as
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

        def_tagging_real = self.def_model.forward(self.dataset.data)
        real_loss = self.def_loss_fn(def_tagging_real, valid)
        #TODO generate the currect tagging for the network with attack node
        currect_tagging_attacked = something(valid, attack_output)

        # here we use the `fake` labels when training the discriminator
        def_tagging_attacked = self.def_model.forward(attack_output)
        fake_loss = self.def_loss_fn(def_tagging_attacked, currect_tagging_attacked)

        def_loss = ((1 - self.lam) * real_loss + self.lam * fake_loss)
        self.def_optimzer.zero_grad()
        def_loss.backward()
        self.def_optimzer.step()
        return att_loss, def_loss

    def test(self):
        with torch.no_grad():
            something()

        return


    def evaluate(self):

        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(range(len(self.gener_costs)), self.gener_costs, label='Generator loss')
        ax1.plot(range(len(self.discr_costs)), self.discr_costs, label='Discriminator loss')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.legend()
        # Set scond x-axis
        ax2 = ax1.twiny()
        newlabel = list(range(NUM_EPOCHS + 1))
        iter_per_epoch = len(train_loader)
        newpos = [e * iter_per_epoch for e in newlabel]
        ax2.set_xticklabels(newlabel[::10])
        ax2.set_xticks(newpos[::10])
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 45))
        ax2.set_xlabel('Epochs')
        ax2.set_xlim(ax1.get_xlim())

