import pandas as pd

import settings as st
import pdb


def preprocess(input_dict, imputer):
    # Categorical
    for key in st.CATEGORIES.keys():
        if key in st.CATEGORICAL_FEATURES:
            for category in st.CATEGORIES[key]:
                if input_dict[key].upper() == category:
                    input_dict[key + '_' + str(category)] = 1
                else:
                    input_dict[key + '_' + str(category)] = 0
    for key in st.CATEGORIES.keys():
        del input_dict[key]
    X = pd.DataFrame(input_dict, index=[0])[st.FINAL_FEATURES]
    # Numerical
    for var in st.NUMERICAL_FEATURES:
        X[var] = X[var].fillna(0)

    X[st.NUMERICAL_FEATURES] = imputer.transform(
        X[st.NUMERICAL_FEATURES])
    
    return X


def predict_buy(input, model, imputer, threshold):
    X = preprocess(input, imputer)
    
    score = model.predict_proba(X)[0][1]
    score = round(float(score), 6)

    if score >= threshold:
        buy_car = 1
        label = 'Will Buy Car'
    else:
        buy_car = 0
        label = 'Will not Buy Car'
    return {
        'probability': score,
        'label': label,
        'buy_car': buy_car
    }
