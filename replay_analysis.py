#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scapy 
from scapy.all import *
import numpy as np
import pickle


# In[3]:


replay=rdpcap('Desktop/ae_replay.pcap')
indoor=rdpcap('Desktop/ae_1722.pcap')


# In[4]:


y=np.zeros((len(indoor),1))
replay_packets=[]


# In[5]:


for i in replay: 
    b = raw(i)
    k=b
    replay_packets.append(list(k))


# In[6]:


count=-1
for i in indoor:
    if(i[0].type==33024):
        count+=1
        b = raw(i)
        k=b
        for j in replay_packets:
            if(list(k)==j):
                y[count]=1
                break
            else:
                y[count]=0


# In[7]:


np.unique(y,return_counts=True)


# In[8]:


indoor_packets=[]
for i in indoor: 
    b = raw(i)
    k=b
    indoor_packets.append(list(k))


# In[9]:


indoor_packets=np.array(indoor_packets)
replay_packets=np.array(replay_packets)


# In[12]:


indoor_packets


# In[111]:


k=indoor_packets[0:160]
from matplotlib import pyplot as plt
plt.figure(figsize=[20,20])
plt.imshow(k, interpolation='nearest')
plt.show()


# In[43]:


dtotal=np.zeros((149600,66))
ytotal=np.zeros((149600,1))


# In[44]:


for i in range((1496)):
    dtotal[i*100:(i+1)*100]=np.concatenate((indoor_packets[i*60:(i+1)*60],replay_packets))
    ytotal[i*100:(i+1)*100]=np.concatenate((np.zeros((60,1)),np.ones((40,1))))


# In[45]:


dtotal.shape


# In[46]:


k=dtotal[0:160]
from matplotlib import pyplot as plt
plt.figure(figsize=[20,20])
plt.imshow(k, interpolation='nearest')
plt.show()


# In[47]:


with open("xa1.pickle","wb") as f:
    pickle.dump(dtotal,f)
with open("ya1.pickle","wb") as f:
    pickle.dump(ytotal,f)


# In[48]:


with open("xn1.pickle","wb") as f:
    pickle.dump(indoor_packets,f)
with open("yn1.pickle","wb") as f:
    pickle.dump(np.zeros((len(indoor_packets),1)),f)


# # Windows

# In[49]:


count=0
for i in range(dtotal.shape[0]):
    if(i+44<dtotal.shape[0]):
        count+=1


# In[50]:


d=np.zeros((count,45,66),dtype='uint8')
for i in range(d.shape[0]):
    d[i]=dtotal[i:i+45].astype(int)


# In[51]:


y1=np.zeros((count,1),dtype='uint8')
for i in range(d.shape[0]):
    y1[i]=ytotal[i+44]


# In[52]:


count=0
for i in range(indoor_packets.shape[0]):
    if(i+44<indoor_packets.shape[0]):
        count+=1


# In[53]:


d1=np.zeros((count,45,66),dtype='uint8')
for i in range(d1.shape[0]):
    d1[i]=indoor_packets[i:i+45].astype(int)


# In[54]:


y2=np.zeros((count,1),dtype='uint8')
for i in range(d1.shape[0]):
    y2[i]=np.zeros((1,1))


# # CNN

# In[ ]:


import tensorflow as tf

X=np.concatenate((d,d1))
Y=np.concatenate((y1,y2))

print(np.unique(Y,return_counts=True))


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.1, stratify=Y,random_state=10)


# In[ ]:


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')


# In[105]:


model=tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(20,input_shape=[X.shape[1],X.shape[2]],kernel_regularizer="L2",return_sequences=True))
model.add(tf.keras.layers.LSTM(5))
#model.add(tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=0.001))
#model.add(tf.keras.layers.Dropout(rate=0.3))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


# In[106]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=["Recall"])


# In[107]:


model.fit(X_train,Y_train,validation_data=(xtest,ytest),epochs=20,batch_size=64,callbacks=[callback])


# In[84]:


model.evaluate(X_val,Y_val)


# In[91]:


y_pred=model.predict_classes(X_val)


# In[92]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_val,y_pred)
print(cm.ravel())


# In[93]:


TN,FP,FN,TP=cm.ravel()
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
F1=2*Precision*Recall/(Precision+Recall)
print("F1 "+str(F1))
print("Precision "+str(Precision))
print("Recall "+str(Recall))


# In[94]:


with open("xtest.pickle","rb") as f:
    xtest=pickle.load(f)
with open("ytest.pickle","rb") as f:
    ytest=pickle.load(f)


# In[108]:


y_pred=model.predict_classes(xtest)


# In[109]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,y_pred)
print(cm.ravel())


# In[110]:


TN,FP,FN,TP=cm.ravel()
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
F1=2*Precision*Recall/(Precision+Recall)
print("F1 "+str(F1))
print("Precision "+str(Precision))
print("Recall "+str(Recall))


# In[ ]:




