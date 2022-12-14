#!/usr/bin/env python
# coding: utf-8

# # Prédiction de budget

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


##lire et afficher notre dataset
df = pd.read_csv("./movies.csv")
df.head()


# In[15]:


##visualiser les valeurs null de notre dataset
plt.figure()
sns.heatmap(df.isna(),cbar=False)


# In[16]:


##Afficher la sommes des valeurs null pour chaque attribut
df.isnull().sum()


# In[17]:


##Afficher les informations de notre dataset
df.info()


# ## Nettoyage des données

# In[18]:


##Création du fonction "dropNull" qu'on va l'utiliser pour supprimer les lignes qui contiennent des valeurs manquantes 
##pour un nom d'attribut donné:
def dropNull(dataframe,col_name):
    movies = dataframe.dropna(subset=[col_name])
    return movies


# In[19]:


df = dropNull(df,'rating')
df = dropNull(df,'released')
df =  dropNull(df,'writer')
df = dropNull(df,'star')
df = dropNull(df,'country')
df = dropNull(df,'company')
df.info()


# In[20]:


##Calcul de median pour les attributs score, votes, budget, runtime et gross
score_m = df["score"].median()
votes_m = df["votes"].median()
budget_m = df["budget"].median()
runtime_m = df["runtime"].median()
gross_m = df["gross"].median()

print(score_m)
print(budget_m)
print(votes_m)
print(runtime_m)
print(gross_m)


# In[21]:


##Remplacer les valeurs null des attributs score, votes, budget, runtime et gross avec leurs median
df['score'][df['score'].isnull()]=score_m
df['votes'][df['votes'].isnull()]=votes_m
df['budget'][df['budget'].isnull()]=budget_m
df['runtime'][df['runtime'].isnull()]=runtime_m
df['gross'][df['gross'].isnull()]=gross_m


df.info()


# In[22]:


##Verifier que notre dataset est bien nettoyé
df.isnull().sum()


# ## visualiser les données

# In[23]:


#Compter le nombre de répétition de chaque Genre
from collections import Counter
genre_raw = df['genre'].to_list()
genre_df = pd.DataFrame.from_dict(Counter(genre_raw), orient = 'index').rename(columns = {0:'Count'})
genre_df


# In[24]:


##Afficher la distribution des genres 
##les trois genres de films les plus produits sont comdey , action , drama
import plotly.express as px

fig = px.pie(data_frame = genre_df,
             values = 'Count',
             names = genre_df.index,
             color_discrete_sequence = px.colors.qualitative.Safe)

fig.update_traces(textposition = 'inside',
                  textinfo = 'label+percent',
                  pull = [0.05] * len(genre_df.index.to_list()))

fig.update_layout(title = {'text':'Distribution des genres'},
                  legend_title = 'Gender',
                  uniformtext_minsize=13,
                  uniformtext_mode='hide',
                  font = dict(
                      family = "Courier New, monospace",
                      size = 18,
                      color = 'black'
                  ))


fig.show()


# In[25]:


plt.bar(df['year'].unique(),df['year'].value_counts())
plt.xticks(rotation=90)
plt.xlabel('years')
plt.ylabel('count')
plt.show()
##Afficher le nombre des films par rapport les années de production


# In[26]:


result=df.groupby('year').sum()
result['budget']


# In[27]:


years = [year for year , df in df.groupby('year')]
plt.bar(years,result['budget'])
plt.xticks(rotation=90)
plt.xlabel('years')
plt.ylabel('budget')
plt.show()

##Affichage d'augmentation de budget par rapport aux années de producions
##Nous remarquons que 2016 est l'années avec le budget le plus élevé


# In[28]:


res=df.groupby('genre').sum()
res


# In[29]:


genre = [genre for genre , df in df.groupby('genre')]
plt.bar(genre,res['budget'])
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('budget')
plt.show()
##budget par rapport au genre 
##nous remarquons que le budget d'un film action est très élevé par rapport au autres genres 


# In[30]:


plt.bar(df['genre'],df['score'])
plt.xticks(rotation=90)
plt.xlabel('genre')
plt.ylabel('score')
plt.show()


# In[31]:


##Utiliser la bib OrdinalEncoder pour encoder les variables catégoriques
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
df["genre"] = ord_enc.fit_transform(df[["genre"]])
df["country"] = ord_enc.fit_transform(df[["country"]])
df["company"] = ord_enc.fit_transform(df[["company"]])
df["rating"] = ord_enc.fit_transform(df[["rating"]])
df["director"] = ord_enc.fit_transform(df[["director"]])
df["writer"] = ord_enc.fit_transform(df[["writer"]])
df["star"] = ord_enc.fit_transform(df[["star"]])



df.head(11)


# In[32]:


##Création de dataset de prediction
cols = ["year","score","votes","gross","budget","runtime","genre","country","company","rating","star"]; 
train_data = df
train_data = train_data[cols] 
train_data.head()


# In[33]:


##Utilisation de bib StandardScaler pour normaliser les valeurs de notre dataset pour faciliter la prediction et avoir des bonnes resultas 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data);
train_data = pd.DataFrame(train_data)
train_data.head()


# In[34]:


##Dataset training 
from sklearn.model_selection import train_test_split
x=train_data.drop(4 , axis=1)
y=train_data[4]
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# In[25]:


##prediction avec LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score , recall_score , accuracy_score , r2_score 

model = LinearRegression()
model.fit(x_train,y_train)

pred = model.predict(x_test)

print(r2_score(y_test, pred))
print(model.score(x_test, y_test))


# In[32]:


##Affichage de resultat de prediction par rapport nos données actuelles 
def plotPred(y_true,y_pred):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, c='crimson')
    plt.yscale('log')
    plt.xscale('log')

    p1 = max(max(y_pred), max(y_true))
    p2 = min(min(y_pred), min(y_true))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()


# In[33]:


plotPred(y_test,pred)


# In[39]:


def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))
    return mape


# In[36]:


import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, BayesianRidge,ARDRegression
from sklearn.metrics import r2_score, mean_squared_error
##prediction avec Ridge
model1 = Ridge(alpha=0.00001)
# training
model1.fit(x_train, y_train)
# prediction
pred2 = model1.predict(x_test)
# evaluation
print(model1)
print('  Train R2 = ', '%.3f' %r2_score(y_test,pred2))
print('  Train RMSE = ', '%.3E' %np.sqrt(mean_squared_error(y_test,pred2)))
print('  Train MAPE = ', '%.0f' %MAPE(y_test,pred2))
     


# In[40]:


##prediction avec XGBRegressor
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(x_train, y_train)
preds = my_model.predict(x_test)
print('  Train R2 = ', '%.3f' %r2_score(y_test, preds))
print('  Train RMSE = ', '%.3E' %np.sqrt(mean_squared_error(y_test,preds)))
print('  Train MAPE = ', '%.0f' %MAPE(y_test,preds))


# In[38]:


plotPred(y_test,preds)


# In[41]:


# choix de modele
model2 = RidgeCV()
# training
model2.fit(x_train, y_train)
# prediction
y_train_pred = model2.predict(x_test)
# evaluations
print(model2)
print('  Train R2 = ', '%.3f' %r2_score(y_test, y_train_pred))
print('  Train RMSE = ', '%.3E' %np.sqrt(mean_squared_error(y_test,y_train_pred)))
print('  Train MAPE = ', '%.0f' %MAPE(y_test,y_train_pred))


# In[42]:


# choix de modele
model3 = Lasso(alpha=0.01)
# training
model3.fit(x_train, y_train)
# prediction
y_train_pred = model3.predict(x_test)
# evaluations
print(model3)
print('  Train R2 = ', '%.3f' %r2_score(y_test, y_train_pred))
print('  Train RMSE = ', '%.3E' %np.sqrt(mean_squared_error(y_test,y_train_pred)))
print('  Train MAPE = ', '%.0f' %MAPE(y_test,y_train_pred))


# In[168]:


from sklearn.model_selection import GridSearchCV
lr=LinearRegression()
lr_params={}
lr_grid=GridSearchCV(lr,lr_params,cv=10,verbose=10)
lr_grid.fit(x_train,y_train)
lr_score=lr_grid.cv_results_
print(lr_score)
preds=lr_grid.predict(x_test)
print(preds)
print('  Train R2 = ', '%.3f' %r2_score(y_test, preds))
print('  Train RMSE = ', '%.3E' %np.sqrt(mean_squared_error(y_test,preds)))


# # Nous avons utilisé 6 modèles de prédiction dans ce projet et le meilleur score que nous avons obtenu est 0.761 avec le modèle XGBRegressor.

# In[35]:


import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf


# In[93]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])


# In[95]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[100]:


hist = model.fit(x_train,y_train,epochs=20,validation_data=(x_test, y_test))
model.evaluate(x_test,y_test,verbose=2)


# In[ ]:




