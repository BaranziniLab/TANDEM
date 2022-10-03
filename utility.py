from paths import *
import pandas as pd
import numpy as np
import joblib
import time
import sys
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


temporal_test_data = np.load(TEMPORAL_TEST_DATA_PATH)
non_temporal_test_data = np.load(NON_TEMPORAL_TEST_DATA_PATH)
train_metadata = pd.read_csv(TRAIN_METADATA_PATH)
test_metadata = pd.read_csv(TEST_METADATA_PATH)

def set_model_hyperparams():
    global n_trees, max_depth, n_cores, n_epoch, n_batch, lr
    n_cores = 8
    n_trees = 2000
    max_depth = 50
    n_epoch = 100
    n_batch = 10
    lr = 1e-4

def get_tandem_train_data(temporal_model, non_temporal_model, train_metadata):
    temporal_train_data = np.load(TEMPORAL_TRAIN_DATA_PATH)
    non_temporal_train_data = np.load(NON_TEMPORAL_TRAIN_DATA_PATH)
    y_score_temporal_train = get_predictions(temporal_model, temporal_train_data)
    del(temporal_train_data)
    y_score_non_temporal_train = get_predictions(non_temporal_model, non_temporal_train_data)
    del(non_temporal_train_data)
    train_df = pd.DataFrame(list(zip(y_score_non_temporal_train, y_score_temporal_train, train_metadata.label.values)), columns=["y_score_non_temporal", "y_score_temporal", "label"])
    train_df["y_score_non_temporal_percentile"] = train_df.y_score_non_temporal.apply(lambda x:stats.percentileofscore(train_df.y_score_non_temporal, x))/100
    train_df["y_score_temporal_percentile"] = train_df.y_score_temporal.apply(lambda x:stats.percentileofscore(train_df.y_score_temporal, x))/100        
    train_df.to_csv(TRAIN_DATA_SCORE_PATH, index=False, header=True)
    return train_df

def get_tandem_test_data(temporal_model, non_temporal_model, temporal_test_data, non_temporal_test_data, test_metadata):
    if os.path.exists(TRAIN_DATA_SCORE_PATH):
        train_df = pd.read_csv(TRAIN_DATA_SCORE_PATH)
    else:        
        train_df = get_tandem_train_data(temporal_model, non_temporal_model, train_metadata)
    y_score_temporal_test = get_predictions(temporal_model, temporal_test_data)
    y_score_non_temporal_test = get_predictions(non_temporal_model, non_temporal_test_data)
    test_df = pd.DataFrame(list(zip(y_score_non_temporal_test, y_score_temporal_test, test_metadata.label.values)), columns=["y_score_non_temporal", "y_score_temporal", "label"])
    test_df["y_score_temporal_percentile"] = test_df.y_score_temporal.apply(lambda x:stats.percentileofscore(train_df.y_score_temporal, x))/100
    test_df["y_score_non_temporal_percentile"] = test_df.y_score_non_temporal.apply(lambda x:stats.percentileofscore(train_df.y_score_non_temporal, x))/100
    return test_df
        
    
def load_tandem(temporal_model, non_temporal_model, train_flag=False):
    if train_flag:
        print("Selected to train TANDEM model")
        set_model_hyperparams()
        train_df = get_tandem_train_data(temporal_model, non_temporal_model, train_metadata)
        test_df = get_tandem_test_data(temporal_model, non_temporal_model, temporal_test_data, non_temporal_test_data, test_metadata)
        X_train = train_df[["y_score_non_temporal_percentile", "y_score_temporal_percentile"]].values
        y_train = train_df["label"].values
        X_test = test_df[["y_score_non_temporal_percentile", "y_score_temporal_percentile"]].values
        y_test = test_df["label"].values
        neg, pos = np.bincount(y_train)
        total = neg + pos
        initial_bias = np.log([pos/neg])
        weight_for_0 = (1 / neg)*(total)/2.0
        weight_for_1 = (1 / pos)*(total)/2.0
        class_weight = {0: weight_for_0, 1: weight_for_1}
        input_dim = X_train.shape[-1]
        output_bias = tf.keras.initializers.Constant(initial_bias)        
        model = Sequential()
        model.add(Dense(1, activation='sigmoid', input_shape=(input_dim, ), bias_initializer=output_bias))
        model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss = "binary_crossentropy", metrics = ["AUC"])
        early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor='val_auc', 
                            verbose=1,
                            patience=10,
                            mode='max',
                            restore_best_weights=True)
        train_history = model.fit(X_train, 
                          y_train,
                          epochs = n_epoch,
                          batch_size = n_batch,
                          callbacks = [early_stopping],
                          validation_data = (X_test, y_test),
                          class_weight = class_weight)
        del(X_train)
        del(X_test)
        return model
    else:
        print("Selected to use pre-trained TANDEM model")
        print("Loading pre-trained TANDEM model")
        return load_model(TANDEM_MODEL_PATH)
        
        
    

def load_model_type(model_type, train_flag=False):
    if train_flag:
        set_model_hyperparams()
        model = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, n_jobs=n_cores, bootstrap=True, max_features='sqrt', warm_start=False, verbose=1)
        if model_type=="non_temporal":
            print("Selected to train non-temporal model")
            print("Loading training data for non-temporal model...")
            X = np.load(NON_TEMPORAL_TRAIN_DATA_PATH)
            print("Training data is loaded!")
        elif model_type=="temporal":
            print("Selected to train temporal model")
            print("Loading training data for temporal model...")
            X = np.load(TEMPORAL_TRAIN_DATA_PATH)
            print("Training data is loaded!")
        else:
            print("model type is not given correctly. your options are 'temporal' or 'non_temporal'. Exiting now!")
            sys.exit(1)
        y = train_metadata.label.values
        sample_wt = compute_sample_weight(class_weight='balanced', y=y)
        start_time = time.time()
        model.fit(X=X, y=y, sample_weight=sample_wt)
        del(X)
        tot_time = round((time.time()-start_time)/(60*60), 2)
        print('{} model training is completed in {} hrs'.format(model_type, tot_time))
        return model
    else:
        if model_type=="non_temporal":
            print("Selected to use pre-trained non-temporal model")
            print("Loading pre-trained non-temporal model")
            try:
                model = joblib.load(NON_TEMPORAL_MODEL_PATH)
                return model
            except:
                print("Cannot load pre-trained {} model since it is not found in the specified path. Exiting now!".format(model_type))
                sys.exit(1)
        elif model_type=="temporal":
            print("Selected to use pre-trained temporal model")
            print("Loading pre-trained temporal model")        
            try:
                model = joblib.load(TEMPORAL_MODEL_PATH)
                return model
            except:
                print("Cannot load pre-trained {} model since it is not found in the specified path. Exiting now!".format(model_type))
                sys.exit(1)
        else:
            print("model type is not given correctly. options are 'temporal' or 'non_temporal'. Exiting now!")
            sys.exit(1)
        
                    
def get_predictions(model, test_data):
    model.verbose = 0
    y_score = model.predict_proba(test_data)[:, 1]
    return y_score 


def get_tandem_predictions(tandem_model, temporal_model, non_temporal_model, temporal_test_data, non_temporal_test_data, test_metadata):
    test_df = get_tandem_test_data(temporal_model, non_temporal_model, temporal_test_data, non_temporal_test_data, test_metadata)
    X_test = test_df[["y_score_non_temporal_percentile", "y_score_temporal_percentile"]].values
    y_test = test_df["label"].values
    y_score = tandem_model.predict(X_test)
    optThresh = get_optimal_threshold(y_test, y_score)
    return y_score, optThresh 

def get_optimal_threshold(y_label, y_score):
    fpr, tpr, thresholds = roc_curve(y_label, y_score)
    specificity_arr = 1 - fpr
    gmean = np.sqrt(tpr * specificity_arr)
    index_sel = np.argmax(gmean)
    optThresh = thresholds[index_sel]
    return optThresh
            
def get_auc(y_label, y_score):
    return roc_auc_score(y_label, y_score)

def get_tandem_single_patient_prediction(patient_index, optThresh, temporal_model, non_temporal_model, tandem_model, test_metadata):
    if os.path.exists(TRAIN_DATA_SCORE_PATH):
        train_df = pd.read_csv(TRAIN_DATA_SCORE_PATH)
    else:        
        train_df = get_tandem_train_data(temporal_model, non_temporal_model, train_metadata)
    patient_sel = 1
    label_sel = test_metadata.iloc[patient_index].label
    temp_prediction = get_predictions(temporal_model, np.expand_dims(temporal_test_data[patient_index,:], 1).transpose())
    non_temp_prediction = get_predictions(non_temporal_model, np.expand_dims(non_temporal_test_data[patient_index,:], 1).transpose())
    temp_prediction_nor = stats.percentileofscore(train_df.y_score_temporal, temp_prediction)/100
    non_temp_prediction_nor = stats.percentileofscore(train_df.y_score_non_temporal, non_temp_prediction)/100
    X = [non_temp_prediction_nor, temp_prediction_nor]
    y_score = tandem_model.predict(np.expand_dims(np.array(X),1).transpose())
    prediction = int(y_score>=optThresh)
    if label_sel == 1:
        label_type = "PD"
    else:
        label_type = "Non-PD"
    if prediction == 1:
        prediction_type = "PD"
    else:
        prediction_type = "Non-PD"
    print("********** Patient prediction ***********")
    print("Label : ", label_type)
    print("TANDEM prediction : ", prediction_type)
    print("TANDEM prediction score :", round(y_score[0][0],2))