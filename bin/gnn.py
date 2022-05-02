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
    "message_passing_iterations" : 3,
    "node_nn_dim" : 100,
    "edge_nn_dim" : 100,
    "latent_dim" : 120
}

gtf = []
gene_sets = []
hintfiles = []
anno = ''
graph = None
out = ''
v = 0
quiet = False
numb_features = {'input_nodes' : 160, 'input_edges' : 240}


def get_inputs(input_components, batch_size, shuffle=True): 
    if shuffle:
        np.random.shuffle(input_components)    
    input_train = []
    batch = []
    curr_lens = {'input_nodes' : 0, 'input_edges' : 0, 'target_label' : 0}
    save_comps =[]
    for k, comp in enumerate(input_components):
        if len(save_comps)<2:
            save_comps.append(comp)
        else:
            batch.append(comp)
            curr_lens['input_nodes'] += comp[0]['input_nodes'][0].shape[0]
            curr_lens['input_edges'] += comp[0]['input_edges'][0].shape[0]
            curr_lens['target_label'] += comp[1]['target_label'][0].shape[1]
        if curr_lens['input_nodes'] > batch_size or k == len(input_components)-1:
            if curr_lens['input_edges'] == 0:
                c = save_comps.pop()
                batch.append(c)
                curr_lens['input_nodes'] += c[0]['input_nodes'][0].shape[0]
                curr_lens['input_edges'] += c[0]['input_edges'][0].shape[0]
                curr_lens['target_label'] += c[1]['target_label'][0].shape[1]
            if k == len(input_components)-1:
                for c in save_comps:
                    batch.append(c)
                    curr_lens['input_nodes'] += c[0]['input_nodes'][0].shape[0]
                    curr_lens['input_edges'] += c[0]['input_edges'][0].shape[0]
                    curr_lens['target_label'] += c[1]['target_label'][0].shape[1]
                save_comps = []
            #print(curr_lens)
            input_train.append([{},{}])
            for type in ['input_nodes', 'input_edges']:
                input_train[-1][0].update({type : np.expand_dims(np.concatenate([b[0][type][0] for b in batch]), 0)})
                
            for type in ['incidence_matrix_sender', 'incidence_matrix_receiver']:
                input_train[-1][0].update({type : np.expand_dims(np.zeros((curr_lens['input_nodes'],
                                                                   curr_lens['input_edges']), bool),0)})
                i=j=0
                for b in batch:
                    if b[0][type][0].shape[1] > 0:
                        input_train[-1][0][type][0][i:b[0][type][0].shape[0]+i, j:b[0][type][0].shape[1]+j] = b[0][type][0]
                    i+=b[0][type][0].shape[0]
                    j+=b[0][type][0].shape[1]
                input_train[-1][0][type] = tf.convert_to_tensor(input_train[-1][0][type])
                                           
            type = 'target_label'
            input_train[-1][1].update({type : np.expand_dims(np.zeros((curr_lens['input_nodes'],
                                                               curr_lens['target_label'])),0)})
            i = 0
            j = 1
            for b in batch:
                input_train[-1][1][type][0][i:b[1][type][0].shape[0]+i, 0] = b[1][type][0][:,0]
                input_train[-1][1][type][0][i:b[1][type][0].shape[0]+i, j:b[1][type][0].shape[1]+j-1] = b[1][type][0][:,1:]
                i+=b[1][type][0].shape[0]
                j+=b[1][type][0].shape[1]-1
            batch=[]
            curr_lens = {'input_nodes' : 0, 'input_edges' : 0, 'target_label' : 0}
    return input_train
            

class GNN:
    def __init__(self, cfg=config, weight_class_one=1.):
        self.model = self.make_GNN(config)
        #self.model_nn = self.make_NN(config)
        self.weight_class_one = weight_class_one
        self.cfg=cfg
        self.cee = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.acc = tf.keras.metrics.BinaryAccuracy()
        self.learning_rate = 1e-4      
        
    #def sp_sn_loss(self, y_true, y_pred):
    def get_cds_probability(self, p_list):
        # input list of probabilities
        return np.sum([p_list[i] * np.prod([1-pj for pj in p_list[:i]]) for i in range(len(p_list))])
        
        
        
    def tx_cds_loss(self, y_true, y_pred):
        loss = 0
        # cds[c_key]: first col: true label of cds, second: probability that c is included in output
        """cds = {}
        tx_true = np.array([y_t[0][1] for y_t in y_true])
        for j in range(len(y_true)):
            for c in y_true[j][1:]:
                if c[0] not in cds:
                    cds.update({c[0] : [c[1], []]})
                cds[c[0]][1].append(y_pred[-1][j])
        cds_true = np.array([c[0] for c in cds.values()])
        cds_pred = np.array([self.get_cds_probability(c[1]) for c in cds.values()])"""
        cds_true = tf.math.reduce_max(y_true[0][:,1:],0)
        #cds_true = tf.zeros(y_true.shape[-1], float)
        cds_true = tf.math.subtract(cds_true, 1.0)
        cds_true = tf.math.maximum(cds_true, 0.0)
        
        cds_weights = tf.math.multiply(cds_true, self.weight_class_one-1.0) + 1.0
        tx_weights = tf.math.multiply(y_true[0][:,0], self.weight_class_one-1.0) + 1.0
        for i in range(self.cfg["message_passing_iterations"]):
            cds_prob = tf.math.minimum(y_true[0][:,1:], 1.0)
            cds_prob = tf.math.multiply(cds_prob, tf.expand_dims(y_pred[i][0],1))
            cum_prod = tf.math.multiply(tf.math.subtract(cds_prob, 1), -1)
            cum_prod = tf.math.cumprod(cum_prod, axis=0, exclusive=True)
            cds_pred = tf.reduce_sum(tf.math.multiply(cds_prob, cum_prod), 0)
            loss += tf.reduce_mean(self.cee(y_true[0][:,0], y_pred[i][0]) * tx_weights)
            loss += tf.reduce_mean(self.cee(cds_true, cds_pred) * cds_weights)
        return loss / self.cfg["message_passing_iterations"]
        
        
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
            #loss += tf.reduce_mean(weights * self.cee(y_true_floor, y_true * y_pred[i]))
        #loss += tf.reduce_mean(self.cee(y_true_floor,tf.math.floor(y_pred[-1] + 0.5)) * weights)
        return loss / self.cfg["message_passing_iterations"]

    
    def last_iteration_binary_accuracy(self, y_true, y_pred):
        return self.acc(y_true[0][:,0], y_pred[-1][0])

    """def nn_cee(self, y_true, y_pred):
        loss = 0
        y_true_floor = tf.math.floor(y_true)
        weights = tf.reshape(y_true_floor[0] * (self.weight_class_one-1.) + 1., [-1])
        loss += tf.reduce_mean(weights * tf.reshape(self.cee(y_true_floor, y_pred[-1]),[-1]))
        loss += tf.reduce_mean(weights * self.cee(y_true_floor, y_true * y_pred[-1]))
        return loss"""

    def compile(self, weights=''):
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.model.compile(loss=self.tx_cds_loss,
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
    def make_ff_layer(self, nn_dim):
        phi = keras.Sequential([
                #layers.Dense(config["latent_dim"], activation=tf.keras.layers.LeakyReLU(alpha=0.01)#,\
                                 #kernel_regularizer=tf.keras.regularizers.l1_l2(0.001)
                                #),
            layers.Dense(nn_dim, activation="relu"#,\
                                # kernel_regularizer=tf.keras.regularizers.l1_l2(0.001)
                                ),
            layers.Dense(nn_dim, activation="relu"#,\
                                # kernel_regularizer=tf.keras.regularizers.l1_l2(0.001)
                                )
                #layers.Dense(config["latent_dim"], activation="relu", \
                        #kernel_regularizer=tf.keras.regularizers.l2(0.004)
                                #),
                #layers.Dense(config["latent_dim"], activation="relu"#, \
                    #kernel_regularizer=tf.keras.regularizers.l1_l2(0.00001)
                            #) 
            #tf.keras.layers.LeakyReLU(alpha=0.01)
        ])
        return phi

    """def make_NN(self, config):
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
        return model"""


    def make_GNN(self, config):
        #define the inputs
        #we use a batch size of 1 since we implemented batching
        #by using the compound graph
        epsi= 1e-10
        target_label = []
        V = keras.Input(shape=(None, numb_features['input_nodes']), name="input_nodes", batch_size=1)
        #B = keras.Input(shape=(None, numb_bias_features), name="input_bias", batch_size=1)
        
        E = keras.Input(shape=(None, numb_features['input_edges']), name="input_edges", batch_size=1)
        
        Is = keras.Input(shape=(None,None), name="incidence_matrix_sender",
                          batch_size=1)
        Ir = keras.Input(shape=(None,None), name="incidence_matrix_receiver",
                          batch_size=1)        
        #bias_encoder = self.make_ff_layer(config)
        node_encoder = self.make_ff_layer(config["latent_dim"])
        edge_encoder = self.make_ff_layer(config["latent_dim"])
        node_updater = self.make_ff_layer(config["latent_dim"])
        edge_updater = self.make_ff_layer(config["latent_dim"])
        bias_encoder = self.make_ff_layer(config["latent_dim"])
        bias_updater = self.make_ff_layer(config["latent_dim"])
        node_decoder = layers.Dense(1, activation="sigmoid")
        
        #step 1: encode
        #transform each node (dim=3) and edge (dim=1) to a latent embedding of "latent_dim"
        V_enc = node_encoder(V)
        E_enc = edge_encoder(E)
        bias_concat = tf.concat([tf.reduce_mean(V_enc, 1), tf.reduce_mean(E_enc, 1)], -1)
        bias_enc = tf.expand_dims(bias_encoder(bias_concat),0)
        #bias_enc = tf.expand_dims(bias_encoder(tf.ones((1,30))),0)
        
        #u = bias_encoder(tf.concat([V,E]))
        #bias_enc = bias_encoder(B)
        #V_enc.shape = (1, num_nodes, latent_dim)
        #E_enc.shape = (1, num_edges, latent_dim)



        #step 2: message passing
        for _ in range(config["message_passing_iterations"]):
            
            E_Vs = tf.matmul(Is, V_enc, transpose_a=True)
            E_Vr = tf.matmul(Ir, V_enc, transpose_a=True)
            #update all edges
            E_concat = tf.concat([E_enc, E_Vs, E_Vr], axis=-1)
            E_concat_bias = tf.pad(E_concat, [[0,0],[0,0],[0,bias_enc.shape[-1]]]) +\
                tf.pad(bias_enc, [[0,0],[0,0],[E_concat.shape[-1],0]])
            E_enc = edge_updater(E_concat_bias)
            
            
            #print(E_enc.shape, Is.shape, tf.matmul(Is, E_enc).shape, tf.reduce_sum(Is, -1)[:,None].shape)
            V_Es_1 = tf.math.divide(tf.matmul(Is, E_enc), tf.expand_dims(tf.reduce_sum(Is, -1),-1) + epsi)
            V_Er_1 = tf.math.divide(tf.matmul(Ir, E_enc), tf.expand_dims(tf.reduce_sum(Ir, -1),-1) + epsi)
            
            i_max = tf.expand_dims(tf.math.argmax(Is * tf.reduce_sum(E_enc, -1),-1),-1)
            #print(Is.shape, tf.reduce_sum(E_enc, -1).shape, i_max.shape)
            V_Es = tf.gather_nd(tf.squeeze(E_enc,0), indices=i_max)
            i_max = tf.expand_dims(tf.math.argmax(Ir * tf.reduce_sum(E_enc, -1),-1),-1)
            V_Er = tf.gather_nd(tf.squeeze(E_enc,0), indices=i_max)
            #V_Es = tf.matmul(Is, E_enc)
            #V_Er = tf.matmul(Ir, E_enc)
            
            #update all nodes based on current state and aggregated edge states
            #V_concat = tf.concat([V_enc, V_Es, V_Er, bias_enc], axis=-1)
            #print('B', bias_enc.shape, V_Es.shape)
            V_concat = tf.concat([V_enc, V_Es, V_Er, V_Es_1, V_Er_1], axis=-1)
            V_concat_bias = tf.pad(V_concat, [[0,0],[0,0],[0,bias_enc.shape[-1]]]) +\
                tf.pad(bias_enc, [[0,0],[0,0],[V_concat.shape[-1],0]])
            V_enc = node_updater(V_concat_bias)
            
            bias_concat = tf.concat([tf.squeeze(bias_enc,0), tf.reduce_mean(V_enc, 1), tf.reduce_mean(E_enc, 1)], -1)
            #print('F', bias_concat.shape)
            bias_enc = tf.expand_dims(bias_updater(bias_concat),0)
            #bias_enc = bias_updater(bias_enc)

            #step 3: decode
            target_label.append(node_decoder(V_enc))

        model = keras.Model(inputs=[V, E, Is, Ir],
                            outputs=[layers.Lambda(lambda x: x, name="target_label")(tf.stack(target_label))])
        return model
