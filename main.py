import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
           

class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with median of column.
        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.fill)

def remove_columns_missing_values(df, min_threshold):
    """
    removes the columns with missing values below a certain threshold
    """
    for col in df.columns:
        rate = sum(df[col].notnull())/float(len(df)) * 100
        if rate <= min_threshold:
            df = df.drop(col,1)
    return df

def cleanAndWrite():
    sns.set(style="whitegrid", color_codes=True)
    np.random.seed(sum(map(ord, "categorical")))
    plt.style.use('ggplot')

    print 'reading excel'
    #df_gtd = pd.read_excel('data/gtd_11to14_0615dist.xlsx')
    df_gtd = pd.read_excel('data/globalterrorismdb_0615dist.xlsx')

    print df_gtd.describe()

    # cleaning by removing unwanted columns
    df_gtd = remove_columns_missing_values(df_gtd, 20)
    print len(df_gtd.columns)
    columns_to_drop = df_gtd.columns[df_gtd.columns.map(lambda x: 'txt' in x)]
    df_gtd = df_gtd.drop(columns_to_drop, 1)
    print df_gtd.dtypes[df_gtd.dtypes.map(lambda x: x == 'object')]
    columns_to_drop = ['summary', 'scite1' , 'scite2' , 'scite3' ,\
            'dbsource' , 'provstate', 'location', 'latitude', 'city',\
            'propcomment', 'weapdetail', 'corp1', 'motive', 'target1']
    df_gtd = df_gtd.drop(columns_to_drop, 1)
    print len(df_gtd.columns)
    columns_to_drop = ['INT_LOG' , 'INT_MISC', 'INT_ANY', 'INT_IDEO']
    df_gtd = df_gtd.drop(columns_to_drop,1)
    columns_to_drop = ['longitude','specificity']
    df_gtd = df_gtd.drop(columns_to_drop,1)
    df_gtd = df_gtd.drop('eventid',1)
    print sum(df_gtd.extended) / float(len(df_gtd))
    df_gtd = df_gtd.drop('extended', 1)
    df_gtd = df_gtd.drop(['nwoundus','nkillus'], 1)
    df_gtd = df_gtd.drop(['nwoundte','propextent','nkillter', 'guncertain1', 'nperpcap'], 1)

    # replacing crits
    print df_gtd[['crit1','crit2','crit3']].head()
    group_crit12 = lambda df: 1 if df.crit1 == 1 and df.crit2 == 1 else 0
    df_gtd['crit1andcrit2'] = df_gtd.apply(group_crit12,axis=1)
    group_crit23 = lambda df: 1 if df.crit2 == 1 and df.crit3 == 1 else 0
    df_gtd['crit2andcrit3'] = df_gtd.apply(group_crit23,axis=1)
    group_crit13 = lambda df: 1 if df.crit1 == 1 and df.crit3 == 1 else 0
    df_gtd['crit1andcrit3'] = df_gtd.apply(group_crit13,axis=1)
    group_crit123 = lambda df: 1 if df.crit1 == 1 and df.crit2 == 1 and df.crit3 == 1 else 0
    df_gtd['crit1andcrit2andcrit3'] = df_gtd.apply(group_crit123,axis=1)
    df_gtd = df_gtd.drop(['crit1','crit2','crit3'],1)

    df_gtd_imp = DataFrameImputer().fit_transform(df_gtd)

    #Saving these to pickle
    print df_gtd.to_pickle('df_gtd.pkl')
    print df_gtd_imp.to_pickle('df_gtd_imp.pkl')

def selectRelFeatures(df_train_x, df_train_y, df_test):
    print 'training to select the most relevant features'
    clf = DecisionTreeClassifier()
    clf.fit(df_train_x, df_train_y)
    print 'done'
    for index_feature, feature_rank in enumerate(list(clf.feature_importances_)):
        print (df_train_x.columns[index_feature], feature_rank)
    perc_val = np.percentile(clf.feature_importances_,70)
    index_relevant_features = list()
    print '\nselecting the top 70 percentile'
    for index_feature, feature_rank in enumerate(list(clf.feature_importances_)):
        if feature_rank >= perc_val:
            print (df_train_x.columns[index_feature], feature_rank)
            index_relevant_features.append(index_feature)

    df_train_x = pd.DataFrame(df_train_x[df_train_x.columns[index_relevant_features]])
    #print df_train_x.shape
    df_test = df_test[df_test.columns[index_relevant_features]]
    #print df_test.shape

    return df_train_x, df_test

def main():
    #cleanAndWrite()

    df_gtd_imp = pd.read_pickle('df_gtd_imp.pkl')

    df_test = df_gtd_imp.query("gname == 'Unknown'")
    df_test = df_test.drop('gname',1)

    df_train = df_gtd_imp.query("gname != 'Unknown'")
    df_train_x = pd.DataFrame(df_train)
    df_train_x = df_train_x.drop('gname',1)
    df_train_y = df_train.gname

    print df_train_x.shape
    print df_train_y.shape
    print df_test.shape

    df_train_x, df_test = selectRelFeatures(df_train_x, df_train_y, df_test)

    print 'training model'
    X_train, X_test, y_train, y_test = \
            train_test_split(df_train_x, df_train_y, test_size=0.3, random_state=0)

    clf_rf = RandomForestClassifier(n_estimators=5)
    clf_rf.fit(X_train, y_train)
    y_pred = clf_rf.predict(X_test)
    print accuracy_score(y_test, y_pred)


if __name__ == '__main__':
    main()

