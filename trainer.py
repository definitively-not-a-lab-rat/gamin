import sys, os, errno, paramfile
import numpy as np 
import pandas as pd
import keras.backend as K
from keras.models import load_model, Model
from keras.layers import Input
from utils import denode_shape, ShadowLoss
from networks_architectures_2 import choose_shadow, choose_generator
import tensorflow as tf


class GAMIN_Trainer:
    '''Agent responsible for handling the training process'''
    def __init__(self):
        self.experiment_name = paramfile.experiment_name
        self._ensure_paths()
        
        # Conventions:
        # gen : z -> x
        # shadow, target : x -> y
        # combined : z -> y
        self.target_model = load_model(paramfile.target_path)
        self.data_validation = paramfile.data_validation
        self.x_shape = denode_shape(self.target_model._feed_input_shapes[0])
        self.y_shape = denode_shape(self.target_model._feed_output_shapes[0])
        self.z_shape = paramfile.noise_shape
        self.generator = choose_generator(paramfile.gen_archi, self.x_shape, self.z_shape)
        self.shadow = choose_shadow(paramfile.shadow_archi, self.x_shape, self.y_shape)
        self.combined = self._build_combined_model()
        self.target_value = paramfile.target_value

        # training params
        self.batch_size = paramfile.batch_size #actually a demi-batch size!
        self.x_batch_shape = (self.batch_size,) + self.x_shape
        self.z_batch_shape = (self.batch_size,) + self.z_shape
        self.y_batch_shape = (self.batch_size,) + self.y_shape
        self.nb_epochs = paramfile.nb_epochs
        self.observe = paramfile.observe
        self.s_opt = paramfile.s_opt
        self.s_on_g_opt = paramfile.s_on_g_opt
        self.metrics = self.target_model.metrics_names[1:]
        self.type_categorical = paramfile.type_categorical
        if self.type_categorical:
            self.combined_loss = 'categorical_crossentropy'
            self.combined_metrics = ['categorical_accuracy', 'mae']
        else:
            self.combined_loss = 'mse'
            self.combined_metrics = ['mae']
        
        # Compile models
        self.shadow.trainable = True
        shadloss = ShadowLoss(gamma=paramfile.gamma_k, initial_k=paramfile.init_k)
        self.shadow.compile(loss=shadloss, optimizer= self.s_opt, metrics=self.metrics)
        self.shadow.trainable = False
        self.combined.compile(loss=self.combined_loss, optimizer=self.s_on_g_opt, metrics=self.combined_metrics)
        self.shadow.trainable = True

        # Prepare logs
        self.log_shadow = []
        self.log_combined = []
        self.log_began = []
    

    def _build_combined_model(self):
        z = Input(shape=self.z_shape)
        x = self.generator(z)
        y = self.shadow(x)
        combined = Model(z, y)
        return combined


    def _train_shadow_pass(self):
        '''perform a single forward-pass for training shadow from target'''
        z = np.random.normal(size=self.z_batch_shape)
        x_g = self.generator.predict(z)
        x_s = np.random.normal(size=self.x_batch_shape)

        # Check shapes are in order
        assert x_g.shape == x_s.shape

        # Get actual output
        y_g = self.target_model.predict(x_g)
        y_s = self.target_model.predict(x_s)
        
        # Arrange alternatively for ShadowLoss
        x = np.zeros( (2*self.batch_size,) + self.x_shape )
        y = np.zeros( (2*self.batch_size,) + self.y_shape )
        x[::2], y[::2] = x_s, y_s
        x[1::2], y[1::2] = x_g, y_g

        # Train
        self.log_shadow.append(self.shadow.train_on_batch(x,y))
        self.log_began.append([self.shadow.loss.k, self.shadow.loss.m_global])
        

    def _build_target_output(self, size):
        '''Replicates the target values to build a target for a batch'''
        y = np.vstack([self.target_value for i in range(size)])
        return y


    def _train_combined_pass(self):
        '''Single pass for the combined model (the generator)'''
        z = np.random.normal(size=self.z_batch_shape)
        y = self._build_target_output(self.batch_size)
        self.log_combined.append(self.combined.train_on_batch(z,y))


    def _ensure_paths(self):
        '''Make sure a path exist for each useful location'''
        self.path_outputs = self.experiment_name +'/outputs/'
        self.path_models = self.experiment_name + '/models/'
        self.path_logs = self.experiment_name + '/logs/'
        for path in [self.path_outputs, self.path_models, self.path_logs]:
            try:
                os.makedirs(path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise e
    
    def save_models(self):
        self.combined.save(self.path_models + 'combined.h5')
        self.shadow.save(self.path_models + 'shadow.h5')
        self.generator.save(self.path_models + 'generator.h5')

    def save_logs(self):
        pd.DataFrame(self.log_shadow, columns=self.shadow.metrics_names).to_csv(self.path_logs + 'log_shadow.csv')
        pd.DataFrame(self.log_began, columns=['k','m_glob']).to_csv(self.path_logs + 'log_began.csv')
        pd.DataFrame(self.log_combined, columns=self.combined.metrics_names).to_csv(self.path_logs + 'log_combined.csv')
    
    def score_shadow(self, new_data=None):
        if new_data:
            self.data_validation = new_data
        if self.data_validation != None:
            return self.shadow.evaluate(self.data_validation)
        else:
            return ['No validation data']
    
    def _print_progress(self, end=False):
        '''Prints a loading bar while training is due'''
        if self.toolbar:
            sys.stdout.write("-")
            sys.stdout.flush()
        elif end:
            sys.stdout.write("\n")
            self.toolbar = False
        else:
            toolbar_width=20
            sys.stdout.write("[%s]" % ("." * toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width+1)) #back to beginning of line
            self.toolbar = True
            

    def train(self, new_epochs=None, new_observe=None):
        '''
        Trains for a determined number of epochs. Those hyperparams can be adjusted when calling.
        
        Inputs:
            new_epochs      (None)      remplaces current number of epochs to train
            new_observe     (None)      remplaces current number of observations epochs
        '''
        if new_epochs: nb_epochs = new_epochs
        else: nb_epochs = self.nb_epochs
        if new_observe: observe = new_observe
        else: observe = self.observe

        self.total_epochs = observe + nb_epochs
        self.toolbar = False

        for epoch in range(observe):
            self._train_shadow_pass()
            if (epoch/self.total_epochs) % 0.05 == 0:
                self._print_progress()
        for epoch in range(observe,nb_epochs):
            # shadow pass
            self._train_shadow_pass()
            # combined pass
            self._train_combined_pass()
            # do it twice to compensate the batch size thing with shadowloss
            self._train_combined_pass()
            if (epoch/self.total_epochs) % 0.05 == 0:
                self._print_progress()
        self._print_progress(end=True)

        return 1