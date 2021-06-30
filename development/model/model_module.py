import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers



class LSTM_ED(tf.keras.Model):

    def __init__(self, tot_dim, hidd_dim, rcr_init, reg,
                       drop_rt):

        super(LSTM_ED, self).__init__()

        self.tot_dim = tot_dim
        self.hidd_dim = hidd_dim
        self.rcr_init = rcr_init
        self.reg = reg
        self.drop_rt = drop_rt
        
        self.lstmE = tf.keras.layers.LSTM(self.hidd_dim,
                                         return_sequences=False,
                                         return_state = True,
                                         recurrent_initializer= self.rcr_init,
                                         kernel_regularizer = regularizers.l2(self.reg),
                                         activation = 'sigmoid',
                                         name='Encoder')

        self.lstmD = tf.keras.layers.LSTM(self.hidd_dim, 
                                          return_sequences = True,
                                          recurrent_initializer= self.rcr_init,
                                          kernel_regularizer=regularizers.l2(self.reg),
                                          activation = 'sigmoid',
                                          name ='Decoder') 

        self.drop = tf.keras.layers.Dropout(self.drop_rt)

        self.dense = tf.keras.layers.Dense(self.tot_dim, 
                                           kernel_regularizer=regularizers.l2(self.reg))
                                           

    def __call__(self, inp_e, inp_d,  training=False):

        inp_e = tf.cast(inp_e, tf.float32)
        inp_d = tf.cast(inp_d, tf.float32)

        # encoder
        _, h, c = self.lstmE(inp_e)

        # decoder
        out_d = self.lstmD(inp_d, initial_state= [h, c])

        if training:
            # Drop
            out_d = self.drop(out_d, training=training)

        # Dense
        out = self.dense(out_d)

        return out

