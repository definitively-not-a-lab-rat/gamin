import numpy as np
from keras.optimizers import Adam

# Parameter file for GAMIN v2
experiment_name = 'gamin-adult-0'

# training params
batch_size = 128
nb_epochs = 20000
observe = 0

# Target Parameters
target_path = ''
type_categorical = True
target_value = np.array([0.,1.])
data_validation = '','' # X,y, or None

## Shadow Parameters
shadow_archi = 's_mlpd'
init_k = 0.001
gamma_k = 0.5
s_opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

## Generator Parameters
gen_archi = 'g_mlpdr'
noise_shape = (10,)
s_on_g_opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
