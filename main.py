import librosa
import glob
import soundfile as sf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Extract features (mfcc, chroma, mel) from a sound file

def get_feature( name, mfcc, chroma, mel):

    with sf.SoundFile( name) as s_f:
        X1 = s_f.read(dtype="float32")
        sample_rate=s_f.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X1))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X1, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
    if mel:
            mel=np.mean(librosa.feature.melspectrogram(X1, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

# - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

def load_data(test_size=0.2):
    x2,y=[],[]
    for file in glob.glob("D:\\projects\\Speech_emotion_recognition\\audiofiles\\Actor_*\\*.wav"):
        name=os.path.basename(file)
        emotion=emotions[name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=get_feature(file, mfcc=True, chroma=True, mel=True)
        x2.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x2), y, test_size=test_size, random_state=9)

x2_train,x2_test,y_train,y_test=load_data(test_size=0.25)

print((x2_train.shape[0], x2_test.shape[0]))

print('Features extracted: {x2_train.shape[1]}')
#MAKE mlpclassifier model
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Train
model.fit(x2_train,y_train)
y_pred=model.predict(x2_test)

import pandas as pd

df=pd.DataFrame({'actual ':y_test,"predicted" :y_pred})
print(df)import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#getdata
url = "titanic.csv"
titanic_data=pd.read_csv(url)
print(titanic_data.shape)
print(titanic_data.head(10))

print('no of passenger :',len(titanic_data))
print("Information : \n",titanic_data.info())

print(
    'Null values info in percentage: \n',
     (titanic_data.isnull().sum()/len(titanic_data))*100
    )

#analyze data
#sns.countplot(x='survived',data=titanic_data)
#plt.show()

#sns.countplot(x='survived',hue='sex',data=titanic_data)
#plt.show()
#sns.countplot(x='survived',hue='pclass',data=titanic_data)
#plt.show()
#titanic_data["age"].plot.hist()
#plt.show()

#data wrangling
print(titanic_data.isnull())
titanic_data.drop('cabin',axis=1,inplace=True)
titanic_data.drop('body',axis=1,inplace=True)
titanic_data.drop('home.dest',axis=1,inplace=True)

print(titanic_data.head(10))
titanic_data.dropna(inplace=True)
print("Information : \n",titanic_data.info())
print((titanic_data.isnull().sum()/len(titanic_data))*100)
print(titanic_data.head(),"\n",titanic_data.shape)

sns.heatmap(titanic_data.isnull(),yticklabels=False,cbar=False)
plt.show()

s=pd.get_dummies(titanic_data['sex'],drop_first=True)
p=pd.get_dummies(titanic_data['embarked'],drop_first=True)
pcl=pd.get_dummies(titanic_data['pclass'],drop_first=True)

titanic_data=pd.concat([titanic_data,s,p,pcl],axis=1)
print(titanic_data.head(10),titanic_data.info())
print(titanic_data['sibsp'].head(20))
titanic_data.drop(['sex','pclass','boat','name','ticket','embarked'],axis=1,inplace=True)
print(titanic_data.head(10))



#train

x=titanic_data.drop('survived',axis=1)
y=titanic_data['survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,shuffle=True)
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)

pred=logmodel.predict(x_test)
#print(classification_report(y_test,pred))
print(accuracy_score(y_test,pred))

#GET ACCURACY
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("MODEL ACCURACY IS : ",accuracy)
#get accuracy score
print("PERCENTAGE ACCURACY: {:.2f}%".format(accuracy*100))

