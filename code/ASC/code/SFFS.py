import pandas as pd
import numpy as np
import copy

def SFFS(speech_df, music_df):
    """
    we will not change the value of parameters in this function
    realize sequential floating forward search.

    param:
        speech_df : pd.DataFrame
        music_df: pd.DataFrame
    return:
        features_k
    """
    # features_r, list of features' name(str), is the complete set of features
    features_r = list(speech_df.columns.values)
    # features_k, list of features' name(str), is the features set during the procession
    features_k = []
    # J_k, list of float number, is the values of criterion function over the k'th features set
    J_k = []
    # initialize k
    k = 0

    # first, we find the 2 best features(pathetic, the are '2 best' just for now)
    for i in range(2):
        fea, J = SFS_select(speech_df, music_df, features_k, features_r)
        # we find one, remove it from features_r, append it to features_k
        features_r.remove(fea)
        features_k.append(fea)
        J_k.append(J)
        k = k + 1
    count = 2
    # keep sequential floating forward search running until there has not any feature left in feature_r
    while(len(features_r) != 0):
        print("************************************************************")
        print("features_k has {} member: {}".format(len(features_k), features_k))
        print("J_k has {} member: {}".format(len(J_k), J_k))
        # step 1: select k+1'th feature from remaining feature set, features_r, to form feature set X_(k+1)
        fea_step_1, J_step_1 = SFS_select(speech_df, music_df, features_k, features_r)
        features_k.append(fea_step_1)
        features_r.remove(fea_step_1)
        
        # step 2: find the least significant feature in the set X_(k+1) and conditional exclusion
        fea_step_2, J_step_2 = weakfeature(speech_df, music_df, features_k)
        # if the least significant feature in X_(k+1) is the new feature selected just now, then that's fine
        if fea_step_2 == fea_step_1: 
            J_k.append(J_step_1)
            k = k + 1
            continue

        # but if it isn't, ok, let's roll, rock and roll
        else:
            # first we kick it out and throw it into garbage, I'm kidding, we pick it up then put it in the remaining feature set
            features_k.remove(fea_step_2)
            features_r.append(fea_step_2)
            # ok, now we compute the criterion value after getting rid of that sh*t
            J_step_2_roll = J_step_2 # criterion(speech_df, music_df, features_k)
            # paper said J_step_2_roll must be lager than J(X_k), let's check it out
            if J_step_2_roll < J_k[k-1]:
                raise ValueError("Oh, God! Help me, show me the mercy!")
            else:
                print("F**k God! I'm doing right! I'm the God!")
            if k == 2:
                J_k[k-1] = J_step_2_roll
                continue
            else:
                # Attention, we are heading at step 3
                # find another least significant feature after we have already find one and kick it out in step 2
                J_step_3 = criterion(speech_df, music_df, features_k)
                fea_step_3_weak, J_step_3_weak = weakfeature(speech_df, music_df, features_k)
                # if the criterion value is bigger than J[X_(k-1)] after we kick the new least signif~ feature out,
                # we will kick it out, for really. And pop the J[X_k]
                while(k > 2 and J_step_3_weak > J_k[k-1-1]):
                    features_k.remove(fea_step_3_weak)
                    features_r.append(fea_step_3_weak)
                    k = k - 1
                    J_k.pop()
                    J_step_3 = criterion(speech_df, music_df, features_k)
                    fea_step_3_weak, J_step_3_weak = weakfeature(speech_df, music_df, features_k)
                
                J_k[k-1] = J_step_3

        count = count + 1
    return features_k

def SFS_select(speech_df, music_df, features_k, features_r):
    """
    we will not change the value of parameters in this function
    this function is used to select the most significant feature
    from remaining feature set, features_r, with respect to the 
    set features_k by sequential forward search method

    parameters:
        speech_df : 
        music_df : 
        features_k : list of features' name(string), selected features set
        features_r : list of features' name(string), remaining features set
    
    return:
        fea : string, name of feature which is selected by SFS.
        max_J : float, the highest criterion value with the most significant feature selected
    """
    if len(features_r) == 0:
        raise ValueError("There has not any feature in remaining features set!")

    # initialize the max J value
    max_J = -np.inf
    fea = ''

    # find the most significant feature
    for feature in features_r:
        J = criterion(speech_df, music_df, features_k+[feature])
        if J > max_J:
            max_J = J
            fea = feature

    return fea, max_J

def weakfeature(speech_df, music_df, features):
    """
    we will not change the value of parameters in this function
    this function is to find the least signicicant feature in the given features set
    
    parameters:
        speech_df : pd.DataFrame
        music_df : pd.DataFrame
        features : list of features' name(string), over which we find the least significant feature
    return:
        fea : string, name of feature which is the least significant
        max_J : float, the maximum criterion value of features set with the least significant feature has taken out
    """

    # check the number of features in set
    if len(features) <= 2:
        raise ValueError("Only the selected features set has least 3 feature can be show in weakfeature() function")

    # initialization
    fea = ''
    max_J = -np.inf

    # find the least significant feature, the least significant feature, which means the criterion value is the 
    # biggest when we take it out if we must take one feature out
    for feature in features:
        # note: we copy it!!
        features_copy = copy.deepcopy(features)
        features_copy.remove(feature)
        J = criterion(speech_df, music_df, features_copy)
        if J > max_J:
            max_J = J
            fea = feature

    return fea, max_J

def criterion(speech_df, music_df, features):
    """
    we will not change the value of parameters in this function
    this function is to compute the value of criterion function over a features set
    parameters:
        speech_df : pd.DataFrame, speech data frame
        music_df : same as above
        features : list of features' name(string), features set over which we compute the value of criterion function
    return:
        J : float
    """
    # compute the covariance matrices of speech/music data over given features set
    cov_speech = np.cov(speech_df[features].values, rowvar=False)
    cov_music = np.cov(music_df[features].values, rowvar=False)

    # compute the within-class scatter matrix
    S_w = (cov_speech + cov_music) / 2

    # compute mixture scatter matrix
    data = pd.concat([speech_df[features], music_df[features]], ignore_index=True)
    S_m = np.cov(data.values, rowvar=False)
    if S_m.size == 1 and S_w.size == 1:
        J = S_m / S_w
    else:
        J = np.linalg.det(S_m) / np.linalg.det(S_w)

    return J