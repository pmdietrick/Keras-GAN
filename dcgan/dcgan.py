from __future__ import print_function, division

from keras.datasets import fashion_mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing import image

import pickle
from os import listdir
from os.path import isfile, join
import os
import cv2

import matplotlib.pyplot as plt

import sys

import numpy as np
import pandas as pd

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.num_epochs = 1

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        #input_const was 7 for 28x28 images. For 32x32 images 7 doesn't work (array mismatch), but 8 does work
        input_const = 8
        KERNEL = 3
        STRIDES = 1
        model = Sequential()

        model.add(Dense(128 * input_const * input_const, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((input_const, input_const, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=KERNEL, strides=STRIDES, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=KERNEL, strides=STRIDES, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(32, kernel_size=KERNEL, strides=STRIDES, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=KERNEL, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        
        print(img.shape)

        return Model(noise, img)
      
    def build_discriminator(self):
        KERNEL = 3

        model = Sequential()

        model.add(Conv2D(32, kernel_size=KERNEL, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=KERNEL, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=KERNEL, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=KERNEL, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
        

    # Function to unpickle the dataset
    def unpickle_all_data(self, directory):
    
        # Initialize the variables
        train = dict()
        test = dict()
        train_x = []
        train_y = []
        test_x = []
        test_y = []
    
        # Iterate through all files that we want, train and test
        # Train is separated into batches
        for filename in listdir(directory):
            if isfile(join(directory, filename)):
            
                # The train data
                if 'data_batch' in filename:
                    print('Handing file: %s' % filename)
                
                    # Opent the file
                    with open(directory + '/' + filename, 'rb') as fo:
                        data = pickle.load(fo, encoding='bytes')

                    if 'data' not in train:
                        train['data'] = data[b'data']
                        train['labels'] = np.array(data[b'labels'])
                    else:
                        train['data'] = np.concatenate((train['data'], data[b'data']))
                        train['labels'] = np.concatenate((train['labels'], data[b'labels']))
                # The test data
                elif 'test_batch' in filename:
                    print('Handing file: %s' % filename)
                
                    # Open the file
                    with open(directory + '/' + filename, 'rb') as fo:
                        data = pickle.load(fo, encoding='bytes')
                
                    test['data'] = data[b'data']
                    test['labels'] = data[b'labels']
    
        
        i=0 #sync data and labels
        # Manipulate the data to the propper format
        for image in train['data']:
            if train['labels'][i] == 8: #limit categories; 8 = ships
                train_x.append(np.transpose(np.reshape(image,(3, 32,32)), (1,2,0)))
            i += 1
        train_y = [label for label in train['labels']]
    
        for image in test['data']:
            test_x.append(np.transpose(np.reshape(image,(3, 32,32)), (1,2,0)))
        test_y = [label for label in test['labels']]
    
        # Transform the data to np array format
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
    
        return (train_x, train_y), (test_x, test_y)


    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = fashion_mnist.load_data()
        # Run the function with and include the folder where the data are
        (X_train, _), (_, _) = self.unpickle_all_data(os.getcwd() + '/data/cifar-10-python/cifar-10-batches-py/')

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        #X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        #record accuracy of descriminator and error of generator for each generation
        d_acc = []
        g_losses = []
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            d_acc.append(100*d_loss[1])
            g_losses.append(g_loss)

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
            
        if epoch == self.num_epochs - 1:
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(style='k.')
            plt.xlabel("Generation")
            plt.ylabel("Accuracy")
            plt.title("Discriminator Accuracy")
            plt.scatter(range(len(d_acc)), d_acc, alpha = .1)
            fig.savefig("cifar_generator_output4/acc_scatter_plot.png")
            
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(style='k.')
            plt.xlabel("Generation")
            plt.ylabel("Loss")
            plt.title("Generator Loss")
            plt.scatter(range(len(g_losses)), g_losses, alpha = .1)
            fig.savefig("cifar_generator_output4/g_loss_scatter_plot.png")

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                img = image.array_to_img(gen_imgs[cnt], scale=True)
                axs[i,j].imshow(img)
                axs[i,j].axis('off')
      
                #axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                #axs[i,j].axis('off')
                cnt += 1
        fig.savefig("cifar_generator_output4/cifar_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=dcgan.num_epochs, batch_size=32, save_interval=50)
