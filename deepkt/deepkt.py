import numpy as np
import tensorflow as tf 
from deep_Knowledge_Tracing import data


class DKTModel(tf.keras.Model):
    """The Deep Knowledge Tracing model
    Arguments in __init__:
        nb_features: The number of features in the input. This is equal to 
        one_hot_encoded_skills which is skill_number*2 + SolvingState
        nb_skills: the number of unique skills in the dataset
        hidden_units: Positive integer. The nummber of units of the LSTM Layer.
        dropout_rate: Float between 0 and 1. Fraction of units to drop

    Raises:
        ValueError: In case of mismatch between the provided input data and 
        what the mode expects.
    """

    def __init__(self, nb_features, nb_skills, hidden_units = 100, dropout_rate = 0.2):
        
        inputs = tf.keras.Input(shape=(None,nb_features), name = 'inputs')
        
        x = tf.keras.layers.Masking(mask_value = data.MASK_VALUE)(inputs)
        
        x = tf.keras.layers.LSTM(hidden_units,
                                 return_sequences = True,
                                 dropout = dropout_rate)(x)

        dense = tf.keras.layers.Dense(nb_skills, activation = 'softmax')
        outputs = tf.keras.layers.TimeDistributed(dense, name = 'outputs')(x)

        super(DKTModel, self).__init__(inputs = inputs,
                                       outputs = outputs,
                                       name = "DKTModel"
                                       )

    def compile(self, optimizer, metrics=None):
        """Configures the Model for training
        Arguments:
            optimizer: String (name of optimizer) or optimizer instance.
            See 'tf.keras.optimizers'.
            
            metrics: List of metrics to be evaluated by the model during training and
            testing. Typically you will use 'metrics = ['accuarcy']. To specify different
            metrics for different outputs of a multi output model, you could also pass a dictionary,
            such as 'metrics = [['accuracy'], ['accuracy', 'mse']]

        Raises:
            ValueErros: In case of invalid arguments for 'optimizer' or 'metrics'

        """

        def custom_loss(y_true, y_pred):
            y_true, y_pred = data.get_target(y_true, y_pred)
            return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        super(DKTModel, self).compile(
            loss = custom_loss,
            optimizer = optimizer,
            metrics = metrics,
            experimental_run_tf_function = False)
        
    def fit (self,
            dataset,
            epochs = 1,
            verbose = 1,
            callbacks=None,
            validation_data = None,
            shuffle = True,
            initial_epoch = 0,
            steps_per_epoch = None,
            validation_steps = None,
            validation_freq = 1):
        """Trains the model for a fixed number of epochs(iterations on a dataset).
        Arguments:
            dataset: A tf.data.dataset. Should return a tuple of '(inputs,(skills,targets))'

            epochs: Integer. Number of epochs to train the model. An epoch is an
            iteration over the entire data provided. Note that in conjunction with
            'initial epoch', 'epochs' is to understand as "final epoch".
            The model is not trainder for a number of iterations given by 'epochs', but
            merely until epoch of index 'epochs' is reached

            verbose: 0,1 or 2. Verbosity mode.
            0 = silent
            1= progress bar
            2 = one line per epoch
            Note that the progress bar is not particularly useful when logged to a file,
            so verbose = 2 is recommended when not running ineractively (eg. in a production environment)

            callbacks: List of 'keras.callbacks.Callback' instances.
            List of callbacks to apply during training.
            See 'tf.keras.callbacks'
            https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback

            validation_data: Data on which to evaluate the loss and any model metrics
            at the end of each epoch. The model will not be trained on this data

            shuffle: Boolean (whether to shuffle the training data before each epoch)

            initial_epoch: Integer. Epoch at which to start training 
            (useful for resuming previous training run)

            steps_per_epoch: Integer or None.
            Total number of steps (batches of samples) before declaring one epoch
            finished and starting the next epoch. The default None is equal to
            the number of samples in your dataset divided by the batch size, or
            1 if that cannot be determined. If x is a tf.data.dataset and steps_per_epoch
            is None tje epoch will run until the input dataset is exhausted.

            validation_steps: Only relevant if 'validation_data' is provided. Total number of
            steps (batches of samples) to draw before stopping when performing validation at the
            end of every epoch. If 'validation_steps' is None, validation will runt until the 
            'validation_data' dataset is exhausted.

            validation_freq: Only relevant if validation data is provided. Integer or 'Collection_abc.Container'
            instance e.g.(list,tuple,etc.). If an integer,specifies how many training epochs 
            to run before a new validation run is performed, e.g. 'validation_freq = 2' runs
            validation every 2 epochs. If a Container, specifies the epochs on which to run validation,
            e.g. 'validation_freq = [1,2,10]' runs validation at the end of the 1st, 2nd and 10th epochs.

        Returns:
            A 'History' object. Its 'History.history' attribute is a record of training loss
            values and metrics values at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        Raises:
            RuntimeError: If the model was never compiled
            ValueError: In case of mismatch between provided input data and 
            the mode lexpects.

        
        
        """
        return super(DKTModel, self).fit (x=dataset,
                                            epochs = epochs,
                                            verbose = verbose,
                                            callbacks = callbacks,
                                            validation_data = validation_data,
                                            shuffle = shuffle,
                                            initial_epoch = initial_epoch,
                                            steps_per_epoch = steps_per_epoch,
                                            validation_steps = validation_steps,
                                            validation_freq = validation_freq)
                                    

    def evaluate(self,
                 dataset,
                 verbose = 1,
                 steps = None,
                 callbacks = None):
        """Returns the loss value and metrics values for the model in test mode.
        Computation is done in batches.
        Arguments:
            dataset: tf.data.dataset. Should return a tuple of
            '(inputs,(skills,targets))'.

            verbose: 0 or 1. Verbosity mode
            0 = silent 
            1 = progress bar

            steps: Integer or None.
            Total number of steps (batches of samples) before declaring
            the evaluation round finished. Ignored with the fefault value of None.
            If x is a tf.data.dataset and 'steps' is None, 'evaluate()' will run
            until the dataset is exhausted. This argument is not supported with array inputs.

            callbacks: List of keras.callbacks.Callbakc instances. 
            List of callbacks to apply during evaluation.
            See https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
        Returns:
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs and/or metrics).
            The attribute 'model.metrics_name' will give you the display labels for
            the scalar outputs.

        Raises:
            ValueError: in case of invalid arguments
        """
        return super(DKTModel, self).evaluate(dataset,
                                              verbose = verbose,
                                              steps = steps,
                                              callbacks = callbacks)
    
