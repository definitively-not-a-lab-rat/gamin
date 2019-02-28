import matplotlib, datetime, os, errno, sys
matplotlib.use('Agg') # to use in server environments
import numpy as np 
import pandas as pd
import tensorflow as tf
import keras.backend as K
from trainer import GAMIN_Trainer
#K.clear_session()


################################################################
# This is a workaround for CC (computing provider)
#try:
#    print('setting environement workaround...')
#    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#except:
#    print('setting environment failed')
################################################################



################################################################
# SCRIPT
################################################################
arguments = sys.argv
parameter_file = arguments[1]
if parameter_file[-3:] != '.py':
    # accept either with or without extension (for shell autofill)
    parameter_file += '.py'

# try to work around the "must be from the same graph" error
graph = tf.Graph()
with graph.as_default():
    session = tf.Session()
    K.set_session(session)
    with session.as_default():
        gamin = GAMIN_Trainer()
print('GAMIN initialization OK')
K.set_session(session)
with graph.as_default():
    gamin.train()
gamin.save_models()
print('Models saved.')
gamin.save_logs()
print('Logs saved.')
score = gamin.score_shadow()
print('Surrogate Validation score:', score)
print('Done!')