from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import auc, roc_curve

number_vars=180
resultr = list(rfe.ranking_ <= number_vars)
selected_features = [vars_selected[i] for i, val in enumerate(resultr) if val == 1]
#selected_features=[var for var in selected_features if var[0:3].lower() in ['app','act']]

start_cols = []

def max_pearson(X):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    upper = upper.fillna(0)
    max_pearson=max(upper.max())
    return max_pearson

def gini(y_true, y_probas):
    fpr, tpr, thresholds = roc_curve(y_true, y_probas)
    gini_test = np.absolute(2 * auc(fpr, tpr) - 1)
    return gini_test

def vif(X):
    vif=1
    for i in range(X.shape[1]):
        if list(X)[i]!=intercept_name:
            vif = max(float(variance_inflation_factor(np.asarray(X), i)),vif)
    return vif

def forward_feature_selection(**kwargs):
    # kwargs
    model = kwargs.get('model', None)
    model_b = kwargs.get('model', None)
    X = kwargs.get('X', None)
    y = kwargs.get('y', None)
    X_test = kwargs.get('X_test', None)
    y_test = kwargs.get('y_test', None)
    vif_threshold = kwargs.get('vif_threshold', None)
    max_vars = kwargs.get('max_vars', None)
    backward = kwargs.get('backward', False)
    start_cols = kwargs.get('start_cols', [])
    repeat = kwargs.get('repeat', 1)

    # SMOTE oversampling
    if SMOTE == True:
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)
    # set variables
    df_dict = {}
    model_dict = {}
    global_iter_ = 1
    total_best_score = 0
    model_df = pd.DataFrame(columns = ['iter', 'cols', 'gini_test', 'gini_train', 'delta_gini', 'max_vif', 'max_pearson'])
    iter_ = 1
    print(iter_)
    best_col = 0
    last_best_score = 0
    best_score = 0.01
    best_score_train = 0.01
    # print('global iter: ' + str(global_iter_))
    X = pd.DataFrame(X)
    X_temp = X[start_cols]
    X_temp_test = X_test[start_cols]
    model.fit(X_temp, y)
    y_probs_forward = model.predict_proba(X_temp_test)[:,1]
    score = gini(y_test, y_probs_forward)
    vif_forward = vif(X_temp)
    method = 'forward'
    X_temp_forward = pd.DataFrame(X[start_cols])
    X_temp_backward = pd.DataFrame(X[start_cols])

    for col in list(X.columns) * repeat:
        if (len(X_temp.columns) >= max_vars) | (col in X_temp.columns) | (col == 'Intercept'):
            continue
        # forward
        method = 'forward'
        X_temp_forward = pd.concat([X_temp, X[col]], axis=1)
        X_temp_forward_test = pd.concat([X_temp_test, X_test[col]], axis=1)
        model.fit(X_temp_forward, y)
        y_probs_forward = model.predict_proba(X_temp_forward_test)[:,1]
        score = gini(y_test, y_probs_forward)
        y_probs_forward_train = model.predict_proba(X_temp_forward)[:,1]
        score_train = gini(y, y_probs_forward_train)
        df_dict[f'{global_iter_}_{iter_}'] = X_temp_forward
        # update score if last best score > score and vif < 5 and coefs > 0
        if (score > last_best_score) & (vif(X_temp_forward) < vif_threshold) & ~any([i < 0 for i in model.coef_.tolist()[0][1:]]):
            best_score = score
            best_score_train = score_train
            best_col = col
            vif_forward = vif(X_temp_forward)
            # backward
            if (iter_ > 1) & (backward == True):
                for col_backward in X_temp_forward.columns:
                    if (col == col_backward)  | (col_backward == 'Intercept'):
                        continue
                    X_temp_backward = X_temp_forward.drop(col_backward, axis=1)
                    X_temp_backward_test = X_temp_forward_test.drop(col_backward, axis=1)
                    model_b.fit(X_temp_backward, y)
                    y_probs_backward = model_b.predict_proba(X_temp_backward_test)[:,1]
                    score = gini(y_test, y_probs_backward)
                    y_probs_backward_train = model_b.predict_proba(X_temp_backward)[:,1]
                    score_train = gini(y, y_probs_backward_train)
                    if (score > best_score) & (vif(X_temp_backward) < vif_threshold) & ~any([i < 0 for i in model.coef_.tolist()[0][1:]]):
                        best_score = score
                        best_score_train = score_train
                        col_to_remove = col_backward
                        vif_backward = vif(X_temp_backward)
                        method = 'backward'
                        X_backward = X_temp_backward
                        X_backward_test = X_temp_backward_test
        if best_col is None:
            break
        if (method == 'forward') & (best_score > last_best_score) & (vif(X_temp_forward) < vif_threshold) & ~any([i < 0 for i in model.coef_.tolist()[0][1:]]):
            X_temp = pd.concat([X_temp, X[best_col]], axis=1)
            X_temp_test = pd.concat([X_temp_test, X_test[best_col]], axis=1)
            max_vif = vif_forward
            max_pearson_ = max_pearson(X_temp)
            delta_gini = best_score_train - best_score
            model_df = model_df.append({'iter' : f'{global_iter_}_{iter_}', 'cols' : list(X_temp.columns), 'gini_test' : best_score, 'gini_train':best_score_train, 'delta_gini':delta_gini, 'max_vif':max_vif, 'max_pearson':max_pearson_}, ignore_index = True)
            last_best_score = best_score
            if best_score > total_best_score:
                total_best_score = best_score
                print('Best gini: ', total_best_score, ' max vif: ', max_vif)
        elif (method == 'backward') & (best_score > last_best_score) & (vif(X_temp_backward) < vif_threshold) & ~any([i < 0 for i in model_b.coef_.tolist()[0][1:]]):
            X_temp = X_backward
            X_temp_test = X_backward_test
            max_vif = vif_backward
            max_pearson_ = max_pearson(X_temp)
            delta_gini = best_score_train - best_score
            model_df = model_df.append({'iter' : f'{global_iter_}_{iter_}', 'cols' : list(X_temp.columns), 'gini_test' : best_score, 'gini_train':best_score_train, 'delta_gini':delta_gini, 'max_vif':max_vif, 'max_pearson':max_pearson_}, ignore_index = True)
            last_best_score = best_score
            if best_score > total_best_score:
                total_best_score = best_score
                print('Best gini: ', total_best_score, ' max vif: ', max_vif)
        else:
            pass
        print(iter_)
        iter_ += 1
    model.fit(X_temp, y)
    return model, model_df

# model, model_df = forward_feature_selection(model=LogisticRegression(max_iter=1000), X=X, y=y, X_test=X_test, y_test=y_test,  vif_threshold=4, max_vars = 8, backward = True, repeat=3, start_cols = start_cols)
