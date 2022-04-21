#!/usr/bin/env python3
# ==============================================================
# author: Lars Gabriel
#
# GNN: data structure of the graph neural network of TSEBRA
# ==============================================================
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import matplotlib.pyplot as plt


class ConfigFileError(Exception):
    pass

config = {
    "message_passing_iterations" : 1
    ,
    "latent_dim" : 128
}

gtf = []
gene_sets = []
hintfiles = []
anno = ''
graph = None
out = ''
v = 0
quiet = False

numb_node_features = 39#46
numb_bias_features = 38
numb_edge_features = 24

numb_batches = 150
batch_size = 100
val_size = 0


class GNN:
    def __init__(self, cfg=config, weight_class_one=1.):

        self.model = self.make_GNN(config)
        #self.model_nn = self.make_NN(config)

        self.weight_class_one = weight_class_one
        self.cfg=cfg
        self.cee = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.acc = tf.keras.metrics.BinaryAccuracy()
        self.learning_rate = 1e-3

    def all_iterations_cee(self, y_true, y_pred):
        loss = 0
        y_true_floor = tf.math.floor(y_true)
        weights = tf.reshape(y_true_floor[0] * (self.weight_class_one-1.) + 1., [-1])
        #print(y_true.shape, y_pred.shape, weights.shape)
        for i in range(self.cfg["message_passing_iterations"]):
            # loss on tx level
            loss += tf.reduce_mean(weights * tf.reshape(self.cee(y_true_floor, y_pred[i]),[-1]))
            #print(self.cee(y_true_floor, y_pred[i]).shape)
            # loss on exon level
            loss += tf.reduce_mean(weights * self.cee(y_true_floor, y_true * y_pred[i]))
        #loss += tf.reduce_mean(self.cee(y_true_floor,tf.math.floor(y_pred[-1] + 0.5)) * weights)
        return loss / self.cfg["message_passing_iterations"]

    def last_iteration_binary_accuracy(self, y_true, y_pred):
        return self.acc(tf.math.floor(y_true), tf.math.floor(y_pred[-1]+0.5))

    def nn_cee(self, y_true, y_pred):
        loss = 0
        y_true_floor = tf.math.floor(y_true)
        weights = tf.reshape(y_true_floor[0] * (self.weight_class_one-1.) + 1., [-1])
        loss += tf.reduce_mean(weights * tf.reshape(self.cee(y_true_floor, y_pred[-1]),[-1]))
        loss += tf.reduce_mean(weights * self.cee(y_true_floor, y_true * y_pred[-1]))
        return loss

    def compile(self, weights=''):
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.model.compile(loss=self.all_iterations_cee,
            optimizer=optimizer,
            metrics={"target_label" : self.last_iteration_binary_accuracy})
        if weights:
            self.model.load_weights(weights)

    def compile_nn(self, weights=''):
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.model_nn.compile(loss=self.nn_cee,
            optimizer=optimizer,
            metrics={"target_label" : self.last_iteration_binary_accuracy})
        if weights:
            self.model.load_weights(weights + '_nn')

    def predict(self, input):
        return self.model(input)

    def predict_nn(self, input):
        return self.model_nn(input)

    def train(self, train, val, num_epochs=10, save_path=''):

        history = self.model.fit(train,
            validation_data=val,
                epochs = num_epochs,
                verbose = 1)
        self.model.save_weights(save_path)
        return history

    def train_nn(self, train, val, num_epochs=10, save_path=''):
        history = self.model_nn.fit(train,
            validation_data=val,
                epochs = num_epochs,
                verbose = 1)
        self.model.save_weights(save_path + '_nn')
        return history

    #define a feedforward layer
    def make_ff_layer(self, config):
        phi = keras.Sequential([
                layers.Dense(config["latent_dim"], activation="relu", \
                    kernel_regularizer=tf.keras.regularizers.l1_l2(0.00001)
                            ),
                #layers.Dense(config["latent_dim"], activation="relu", \
                        #kernel_regularizer=tf.keras.regularizers.l2(0.004)
                                #),
                layers.Dense(config["latent_dim"], activation="relu", \
                    kernel_regularizer=tf.keras.regularizers.l1_l2(0.00001)
                            )
        ])
        return phi

    def make_NN(self, config):
        V = keras.Input(shape=(None, numb_node_features), name="input_nodes", batch_size=1)
        phi = keras.Sequential([
                layers.Dense(config["latent_dim"], activation="relu", \
                    kernel_regularizer=tf.keras.regularizers.l2(0.002)
                            ),
                layers.Dense(config["latent_dim"], activation="relu", \
                    kernel_regularizer=tf.keras.regularizers.l1_l2(0.002)
                            ),
                layers.Dense(1, activation="sigmoid")])
        target_label = [phi(V)]

        model = keras.Model(inputs=[V],
                            outputs=[layers.Lambda(lambda x: x, name="target_label")(tf.stack(target_label))])
        return model


    def make_GNN(self, config):
        #define the inputs
        #we use a batch size of 1 since we implemented batching
        #by using the compound graph
        V = keras.Input(shape=(None, numb_node_features), name="input_nodes", batch_size=1)
        #B = keras.Input(shape=(None, numb_bias_features), name="input_bias", batch_size=1)
        E = keras.Input(shape=(None, numb_edge_features), name="input_edges", batch_size=1)
        Is = keras.Input(shape=(None,None), name="incidence_matrix_sender",
                          batch_size=1)
        Ir = keras.Input(shape=(None,None), name="incidence_matrix_receiver",
                          batch_size=1)

        #bias_encoder = self.make_ff_layer(config)
        node_encoder = self.make_ff_layer(config)
        edge_encoder = self.make_ff_layer(config)
        node_updater = self.make_ff_layer(config)
        edge_updater = self.make_ff_layer(config)
        node_decoder = layers.Dense(1, activation="sigmoid")

        #step 1: encode
        #transform each node (dim=3) and edge (dim=1) to a latent embedding of "latent_dim"
        V_enc = node_encoder(V)
        E_enc = edge_encoder(E)
        #bias_enc = bias_encoder(B)
        #V_enc.shape = (1, num_nodes, latent_dim)
        #E_enc.shape = (1, num_edges, latent_dim)

        target_label = []

        #step 2: message passing
        for _ in range(config["message_passing_iterations"]):

            #for each node, sum outgoing and incoming edges (separately)
            V_Es = tf.matmul(Is, E_enc)
            V_Er = tf.matmul(Ir, E_enc)

            #update all nodes based on current state and aggregated edge states
            #V_concat = tf.concat([V_enc, V_Es, V_Er, bias_enc], axis=-1)
            V_concat = tf.concat([V_enc, V_Es, V_Er], axis=-1)
            V_enc = node_updater(V_concat)

            #get states of respective sender and receiver nodes for each edge
            E_Vs = tf.matmul(Is, V_enc, transpose_a=True)
            E_Vr = tf.matmul(Ir, V_enc, transpose_a=True)
            #update all edges
            E_concat = tf.concat([E_enc, E_Vs, E_Vr], axis=-1)
            E_enc = edge_updater(E_concat)

            #step 3: decode
            target_label.append(node_decoder(V_enc))

        model = keras.Model(inputs=[V, E, Is, Ir],
                            outputs=[layers.Lambda(lambda x: x, name="target_label")(tf.stack(target_label))])
        return model
