import keras.backend as K
import numpy as np

class ShadowLoss():
    '''
    intended to imitate BEGAN loss in BORGNE shadow setting
    (where autoencoder architecture is irrelevant ...)
    crafted from issue#4813 and
    github.com/mokemokechicken/keras_BEGAN/blob/master/src/began/training.py
    # Arguments
        k_init: Float; initial k factor
        lambda_k: Float; k learning rate
        gamma: Float; equilibrium factor
    '''
    
    __name__ = 'shadow_loss'
    
    def __init__(self, initial_k=0.001, lambda_k=0.001, gamma=0.5):
        self.lambda_k = lambda_k
        self.gamma = gamma
        self.k_var = K.variable(initial_k, dtype=K.floatx(), name="shadow_k")
        self.m_global_var = K.variable(np.array([0]), dtype=K.floatx(), name="m_global")
        self.updates=[]

    def __call__(self, y_true, y_pred):  # y_true, y_pred shape: (batch_size, nb_class)
        # LET'S MAKE A STRONG HYPOTHESIS: BATCH IS HALF NOISE & HALF GENERATED
        # ORDERED AS  EVEN NUMBERS = NOISE & ODD NUMBERS = GENERATED
        noise_true, generator_true = y_true[:, ::2], y_true[:, 1::2] #even, odd
        noise_pred, generator_pred = y_pred[:, ::2], y_pred[:, 1::2] #even, odd
        loss_noise = K.mean(K.abs(noise_true - noise_pred))
        loss_generator = K.mean(K.abs(generator_true - generator_pred))
        began_loss = loss_noise - self.k_var * loss_generator
        
        # The code from which this is adapted used an update mechanism
        # where DiscriminatorModel collected Loss Function's updates attributes
        # This is replaced here by LossUpdaterModel (hihihi)

        mean_loss_noise = K.mean(loss_noise)
        mean_loss_gen = K.mean(loss_generator)
        
        # update K
        new_k = self.k_var + self.lambda_k * (self.gamma * mean_loss_noise - mean_loss_gen)
        new_k = K.clip(new_k, 0, 1)
        self.updates.append(K.update(self.k_var, new_k))

        # calculate M-Global
        m_global = mean_loss_noise + K.abs(self.gamma * mean_loss_noise - mean_loss_gen)
        m_global = K.reshape(m_global, (1,))
        self.updates.append(K.update(self.m_global_var, m_global))

        return began_loss
    
    @property
    def k(self):
        return K.get_value(self.k_var)

    @property
    def m_global(self):
        return K.get_value(self.m_global_var)

def denode_shape(shape_tup):
        '''Removes all Nones from a shape tuple '''
        while shape_tup[0] == None:
            shape_tup = shape_tup[1:]
        return shape_tup