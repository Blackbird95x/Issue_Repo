import pandas as pd
import tensorflow as tf
import numpy as np


MASK_VALUE = -1.


def load_dataset(csv_File, batch_size=32, shuffle=True):
    #df = dataframe is the Datainput
    df = pd.read_csv(csv_File)
    columns = ['moduleId','solvingState','studentID']

    #Step 1 Check if all neccesary columns are appearing in the dataset if not raise an Error
    for i in range(len(columns)):
        if columns[i] not in df.columns:
            raise KeyError(f"The column with {columns[i]} was not found on {csv_File}")
        #We also check if the solvingState got the right Value that is 1 for correctly solved and 2 for not correctly solved. Thereby 0 refers to not solved.
        if columns[i] == 'solvingState':
            if not (df['solvingState'].isin([1, 2])).all():
                raise KeyError(f"The values of the column {columns[i]} are not correct an must be in range 1 = correct or 2 = not correct")

    #Step 2 Remove Questions without modules or skills. Neccesary if you dont preprocess the data externally
    df.dropna(subset=['moduleId'], inplace=True)

    #Step 3 Remove all Students with answered qeustion under one . Neccesary if you dont preprocces the data externally
    df = df.groupby('studentID').filter(lambda q: len(q)>1).copy()

    #Step 4 Get a feature which allows us to read how much modules are in the dataset
    #We Enumerate the moduleId to a column called skill so we get consecutive numbers on the module
    df['skill'], _ = pd.factorize(df['moduleId'], sort=True)

    #Step 5 To be able to one hot encode the skill later on we have to create a synthetic feature
    df['one_hot_encode_Skill'] = df['skill']*2 + df['solvingState']

    #Step 6 We now convert to a sequence per studentID which will give us an output of the 
    #Students past answers as well as the moduleID to the past answers and the solvings of the 
    #Student to this particular Problem
    seq = df.groupby('studentID').apply(
        lambda r: (
                r['one_hot_encode_Skill'].values[:],
                r['skill'].values[:],
                r['solvingState'].values[:],
        )
    )
    #We see how much unique users are in the Dataset
    nb_users = len(seq)

    #Step 7 Create a Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator = lambda: seq,
        output_types=(tf.int32, tf.int32, tf.float32)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size = nb_users)


    features_depth = df['one_hot_encode_Skill'].max() + 2 
    skill_depth = df['skill'].max() + 1

    dataset = dataset.map(
    lambda feat, skill, label: (
        tf.one_hot(feat, depth=features_depth),
        tf.concat(
            values=[
                tf.one_hot(skill, depth=skill_depth),
                tf.expand_dims(label, -1)
            ],
            axis=-1
        )
    )
    )
    #Step 7 Pad the sequences per batch
    #This however gets the wrong shape and needs to be fixed
    dataset = dataset.padded_batch(
        batch_size = batch_size,
        padding_values = (MASK_VALUE,MASK_VALUE),
        padded_shapes = ([None,None],[None,None]),
        drop_remainder = True
    )

    length = nb_users // batch_size
    return dataset, length, features_depth, skill_depth


def split_dataset(dataset, total_size, test_fraction, val_fraction=None):
    #Step 1 Splitting the dataset in the size we want to and return
    #our split_set and Dataset excluding our split_se
    def split(dataset, split_size):
        split_set = dataset.take(split_size)
        dataset = dataset.skip(split_size)
        return dataset, split_set
    #Step 2 We check if there are any unsuitable values for our
    #fractions for test and validation
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0,1")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1")
    #Step 3 Setting up the test and training size
    test_size = np.ceil(test_fraction * total_size)
    train_size = total_size - test_size
    #Check if one of both sizes are zero we raise an error because 
    #they are both need some values to train and test
    if test_size == 0 or train_size == 0:
        raise ValueError(
            "The train and/or the test size are equal to 0 but there has to be at least one element in each of them"
        )
    #Step 4 Create a train and test set. We therefore split the set 
    #with the Function which is defined at the top of the function split_dataset
    train_set, test_set = split(dataset, test_size)
    val_set = None
    #If there is a val_fraction we will set it here
    if val_fraction:
        val_size = np.ceil(train_size * val_fraction)
        train_set, val_set = split(train_set, val_size)

    return train_set, test_set, val_set

def get_target(y_true,y_pred):
    #Get skills and labels from y_true
    mask = 1. - tf.cast(tf.equal(y_true,MASK_VALUE), y_true.dtype)
    y_true = y_true * mask
    
    skills, y_true = tf.split(y_true, num_or_size_splits = [-1, 1], axis = -1)

    y_pred = tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    
    return y_true, y_pred