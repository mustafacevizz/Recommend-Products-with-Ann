
#import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


dataset = pd.read_excel('data.xlsx')
dataset.fillna(dataset.mean(), inplace=True)    #Boş hücreleri bulunduğu kolonun ortalamasıyla dolduruyor

X = dataset.iloc[:, :-1].values     #Xler datasetteki predict etmek istediğim kolon için kullanacağım bölüm satın alınan ürün
y = dataset.iloc[:, -1].values      #diğer ürün id

y = tf.keras.utils.to_categorical(y)    #Model sonucunu kategorilere ayırır. ürünlerin alınma oranını idleri kategorize eder


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# verinin %20sini teste geri kalanını trainlere ayırdık.

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()   #deeplerningin daha iyi çalışması için sayıları normalize eder
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


ann = tf.keras.models.Sequential()  #Sequential kullanıyoruz

# Adding the input layer
ann.add(tf.keras.layers.Dense(units=103, activation='relu'))    #Xler inputların kaç nöron kullanacağını belirledik
ann.add(tf.keras.layers.Dense(units=103, activation='relu'))


# Adding the output layer
ann.add(tf.keras.layers.Dense(units=103, activation='softmax')) #Yler 103 farklı ürün her biri için olasılık tahmini olması için 103 ünit dedik


ann.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])    #Ana iskelet hangi fonksiyonları kullanacağımı ayarladık

# Training the ANN on the Training set
history = ann.fit(X_train, y_train, batch_size = 32, epochs = 500, validation_data=(X_test, y_test))    #Datayı train ediyoruz. epochs ile doğruluk değerini arttırmaya çalışıyoruz
#32şer şekilde aldık ürünleri
y_pred = ann.predict(X_test)    #Xtest tahminleri y_prede atıyoruz
#y_pred_binary = y_pred.round(0)


#plots
plt.figure(figsize=(14,3))
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'y', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)
plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'y', label = 'Training Acc')
plt.plot(epochs, val_acc, 'r', label = 'Validation Acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#predict something
p = ann.predict(sc.transform([[1]]))    #Product listesindeki hangi idyi yollarsam sonraki ürünün idsini verir
print(p)