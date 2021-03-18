
import streamlit as st

st.write("""# Feature Selection Examples""")

st.write("""### Examples of feature selection using the hermione framework from A3Data""")
st.write("""### https://github.com/A3Data/hermione""")

st.image('vertical_logo.png')

st.write("""## Import:""")

st.write("""### All necessary modules""")

with st.echo():
    import pandas as pd
    import numpy as np

    from ml.data_source.spreadsheet import Spreadsheet
    from ml.preprocessing.preprocessing import Preprocessing
    from ml.preprocessing.feature_selection import FeatureSelector
    from ml.preprocessing.normalization import Normalizer

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVR
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import mutual_info_classif 
    from skfeature.function.similarity_based import fisher_score 
    from sklearn.feature_selection import chi2

st.write("""### The data""")
with st.echo():
    df = Spreadsheet().get_data('train.csv',columns=['Survived','Pclass','Sex','Age'])

st.write(df.head(5))

st.write("""## Variance Threshold Example""")
with st.echo():
    p = Preprocessing()
    f = FeatureSelector('variance', threshold=0.5)
    df = p.clean_data(df)
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X)
    f.transform(X).head(5)

st.write(f.transform(X).head(5))

st.write("""## Information Gain Example""")
with st.echo():
    p = Preprocessing()
    f = FeatureSelector('univariate_kbest', score_func=mutual_info_classif, k=2)
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Chi-square Example (k best)""")
with st.echo():
    p = Preprocessing()
    f = FeatureSelector('univariate_kbest', score_func=chi2, k=2)
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Chi-square Example (percentile)""")
with st.echo():
    p = Preprocessing()
    f = FeatureSelector('univariate_percentile', score_func=chi2, percentile=10)
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Fisher's Score Example Example """)
with st.echo():
    p = Preprocessing()
    f = FeatureSelector('univariate_kbest', score_func=fisher_score.fisher_score, k=2)
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Correlation Coefficient Example """)
with st.echo():
    p = Preprocessing()
    f = FeatureSelector('correlation', threshold=0.9)
    df = p.clean_data(df)
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Mean Absolute Difference (MAD) Example """)
with st.echo():
    f = FeatureSelector('univariate_kbest', score_func=FeatureSelector.mean_abs_diff, k=2)
    p = Preprocessing()
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Dispersion ratio Example """)
with st.echo():
    f = FeatureSelector('univariate_kbest', score_func=FeatureSelector.disp_ratio, k=2)
    p = Preprocessing()
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Forward Feature Selection Example """)
with st.echo():
    f = FeatureSelector('sequential', estimator = DecisionTreeClassifier(), direction='forward')
    p = Preprocessing()
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Backward Feature Selection Example """)
with st.echo():
    f = FeatureSelector('sequential', estimator = DecisionTreeClassifier(), direction='backward')
    p = Preprocessing()
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Exaustive Feature Selection Example """)
with st.echo():
    f = FeatureSelector('exaustive', estimator = DecisionTreeClassifier(), min_features = 2, max_features = 4)
    p = Preprocessing()
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Recursive Feature Elimination Example """)
with st.echo():
    f = FeatureSelector('recursive', estimator = LinearSVC(), n_features_to_select=2)
    p = Preprocessing()
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## LASSO Regularization (L1) Example """)
with st.echo():
    estimator = LogisticRegression(C=1, penalty='l1', solver='liblinear')
    f = FeatureSelector('model', estimator = estimator)
    p = Preprocessing()
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Random Forest Importance Example """)
with st.echo():
    f = FeatureSelector('model', estimator = RandomForestClassifier())
    p = Preprocessing()
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Coefficients Example """)
with st.echo():
    f = FeatureSelector('coefficients', model=LinearSVC(), num_feat = 2)
    p = Preprocessing()
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))

st.write("""## Ensemble Example """)
with st.echo():
    f = FeatureSelector('ensemble', dic_selection={ 'variance': {'threshold' : 0.3}, 'recursive': {'estimator' : LinearSVC(), 'n_features_to_select' : 2}},
                        num_feat = 1)
    p = Preprocessing()
    df = p.clean_data(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X = p.categ_encoding(X)
    f.fit(X,y)
    f.transform(X)

st.write(f.transform(X).head(5))
