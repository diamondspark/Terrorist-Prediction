import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

df_gtd=pickle.load( open( "df_gtd.pkl", "rb" ) )
df_gtd_imp=pickle.load( open( "df_gtd_imp.pkl", "rb" ) )

#df_gtd_imp=df_gtd_imp.drop(['multiple','doubtterr','ransom','property','claimed','suicide','vicinity'],1)
#df_gtd_imp=df_gtd_imp.drop(['success','suicide','claimed','multiple','vicinity','ishostkid','property','ransom'],1) #for 50 percentile
df_gtd_imp=df_gtd_imp.drop(['vicinity','ishostkid','property','ransom','doubtterr','success','suicide','claimed','multiple','attacktype1','weaptype1','nperps'],1) #for 60 percentile
#df_gtd_imp=df_gtd_imp.drop(['success','suicide','claimed','multiple','vicinity','ishostkid','property','ransom'],1) #for 70 percentile
#df_gtd_imp=df_gtd_imp.drop(['success','suicide','claimed','multiple','vicinity','ishostkid','property','ransom'],1) #for 80 percentile

print df_gtd_imp.head(5)


#df_test = df_gtd_imp.query("gname == 'Unknown'")
#df_test = df_gtd_imp.query("nhostkid==-99")
#df_test[df_test['nwound'].apply(np.isnan)]
#df_test = df_test[np.isfinite(df_test['nwound'])]
df_test = df_gtd_imp[pd.isnull(df_gtd_imp['nkill'])]
#df_test=df_gtd_imp[df_gtd_imp['nkill'].apply(np.isreal)]


#df_test = df_test.drop('gname',1)
#df_test = df_test.drop('nhostkid',1)
df_test = df_test.drop('nkill',1)
df_test = df_test.drop('gname',1)


#df_train = df_gtd_imp.query("gname != 'Unknown'")
#df_train = df_gtd_imp.query("nhostkid!=-99")
#df_train = df_gtd_imp.query('(nkill==nkill)')
#df_train = df_gtd_imp.query("df_gtd_imp['nwound'].notnull()")
#df_train[df_train['nwound'].apply(np.isreal)]
#df_train = df_train[np.isfinite(df_train['nwound'])]
df_train = df_gtd_imp[pd.notnull(df_gtd_imp['nkill'])]
#df_train=df_gtd_imp
#df_train.dropna(subset = ['nwound'])
df_train_x = pd.DataFrame(df_train)
#df_train_x = df_train_x.drop('gname',1)
#df_train_y = df_train.gname
#df_train_x = df_train_x.drop('nhostkid',1)
#df_train_y = df_train.nhostkid
df_train_x = df_train_x.drop('nkill',1)
df_train_x=df_train_x.drop('gname',1)
df_train_y = df_train.nkill
#df_train_y[df_train_y.apply(lambda x: type(x) in [int, np.int64, float, np.float64])]



print df_train_x.shape
print df_train_y.shape
print df_test.shape



#df_train_x = pd.DataFrame(df_train_x[df_train_x.columns['iyear','imonth','iday','country','region','targsubtype1','weapsubtype1','nkill']])
"""
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(df_train_x, df_train_y)

for index_feature, feature_rank in enumerate(list(clf.feature_importances_)):
        print (df_train_x.columns[index_feature], feature_rank)

perc_val = np.percentile(clf.feature_importances_,50)
index_relevant_features = list()
for index_feature, feature_rank in enumerate(list(clf.feature_importances_)):
        if feature_rank >= perc_val:
            print (df_train_x.columns[index_feature], feature_rank)
            index_relevant_features.append(index_feature)


df_train_x = pd.DataFrame(df_train_x[df_train_x.columns[index_relevant_features]])

df_train_x.shape
df_train_y.shape

df_test = df_test[df_test.columns[index_relevant_features]]

df_test.shape
"""
#df_train_x = df_train_x[df_train_x.columns['iyear','imonth','iday','country','region','targsubtype1','weapsubtype1','nkill']]
#df_test = df_test[df_test.columns['iyear','imonth','iday','country','region','targsubtype1','weapsubtype1','nkill']]
"""
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(df_train_x, df_train_y)
clf.feature_importances_

data_feature_rank = pd.DataFrame(clf.feature_importances_)
data_feature_rank.columns = ['rank']

sns.barplot(x = df_train_x.columns, y = 'rank' , data = data_feature_rank, order = df_train_x.columns)
"""
"""
from sklearn.tree import DecisionTreeClassifier
clf_dtree = DecisionTreeClassifier()
clf_dtree = clf_dtree.fit(df_train_x, df_train_y)
clf_dtree.score(df_train_x,df_train_y)
"""

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.3, random_state=0)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))



"""
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train) 
#Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001)
print('Coefficients: \n', clf.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((clf.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % clf.score(X_test, y_test))
"""
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
degree=2
model = make_pipeline(PolynomialFeatures(degree), Lasso())
model.fit(X_train, y_train)
print("Residual sum of squares:%.2f"
    % np.mean((model.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(X_test, y_test))
"""
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
degree=2
model = make_pipeline(PolynomialFeatures(degree), Ridge())
model.fit(X_train, y_train)
print("Residual sum of squares:%.2f"
    % np.mean((model.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(X_test, y_test))
"""
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TheilSenRegressor

degree=2
model = TheilSenRegressor(random_state=42)
model.fit(X_train, y_train)
print("Residual sum of squares:%.2f"
    % np.mean((model.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(X_test, y_test))
"""

"""
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train, y_train) 
print('Coefficients: \n', clf.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((clf.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % clf.score(X_test, y_test))
"""

# Plot outputs
#plt.scatter(X_test, y_test,  color='black')
#plt.plot(X_test, regr.predict(X_test), color='blue',linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()

"""
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=5)
clf_rf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
y_pred = clf_rf.predict(X_test)
accuracy_score(y_test, y_pred)

test_y = clf_rf.predict(df_test)

df_test['gname'] = test_y
"""
import pylab
pylab.show()