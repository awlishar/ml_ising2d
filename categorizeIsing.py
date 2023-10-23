import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow  import keras
from ising import IsingModel2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
#------------------------------------------------------------------------------
#  PREPARE TRAINING DATA
#------------------------------------------------------------------------------
print("Preparing training data.")


folder = ""
meta = np.load(folder + "meta.npy")
data = np.load(folder + "data.npy")
energies = np.load(folder + "elist.npy")
temperatures = np.load(folder + "tlist.npy")
categories = np.load(folder + "clist.npy")
magnetizations = np.load(folder +"mlist.npy")

print(f"N={N}, Number of sites={N*N}, Monte Carlo steps={mcsteps}, temp range={t_range}")

# Scale energies to [0,1]
energies /= (mcsteps*N*N)
d_train = data
t_train = temperatures
c_train = categories

f = plt.figure(figsize=(10, 5)) # plot the calculated values

sp =  f.add_subplot(1, 2, 1 )
plt.scatter(temperatures, energies, marker='.', color='r')
plt.xlabel("Temperature (K)")
plt.ylabel("Energy")
plt.axis('tight')
plt.title(folder)
plt.grid(True)

sp =  f.add_subplot(1, 2, 2)
plt.scatter(temperatures, abs(magnetizations)/(mcsteps*N*N), marker='.', color='b')
plt.xlabel("Temperature (K)")
plt.ylabel("Magnetization")
plt.axis('tight')
plt.grid(True)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.4)

plt.savefig("isingdata.png")

#------------------------------------------------------------------------------
#  PREPARE TEST DATA
#------------------------------------------------------------------------------
print("Preparing test data.")
onsagerIsingTc = 1/(1/2*np.log(1+np.sqrt(2)))
generateNewTestData = False

n_test = 1500
t_test = []
d_test = []
c_test = []

if generateNewTestData:
    for i in range(n_test):
        print(i)
        t = random.uniform(onsagerIsingTc-t_range/2, onsagerIsingTc+t_range/2)
        t_test.append(t)
        c_test.append(0 if t < onsagerIsingTc else 1)
        # c_test.append(0 if t < onsagerIsingTc else 1)
        ising = IsingModel2D(N=N, T=t)
        #print(ising)
        ising.runMonteCarlo2(mcsteps)
        d_test.append(ising.lattice)
else:
    c_test = c_train[:n_test]
    t_test = t_train[:n_test]
    d_test = d_train[:n_test]
    c_train = c_train[n_test:]
    t_train = t_train[n_test:]
    d_train = d_train[n_test:]



print(t_test)

#------------------------------------------------------------------------------
#  ML
#------------------------------------------------------------------------------
print("Running ML algorithm.")

x_train, y_train = np.array(d_train), np.array(c_train)
x_test, y_test = np.array(d_test), np.array(c_test)

print(x_train.shape, y_train.shape)

def nature1(N):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(N,N)),
        keras.layers.Dense(100, activation='sigmoid'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

def nature2():
  model = keras.Sequential()

  model.add(Conv2D(filters = 64, kernel_size = (2,2),padding = 'Same',
                  activation ='relu', input_shape = (16,16,1)))

  model.add(Flatten())
  model.add(Dense(64, activation = "relu"))
  model.add(Dropout(0.6))
  model.add(Dense(2, activation='softmax'))
  optimizer = keras.optimizers.Adam()

  model.compile(optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

  return model

model = nature1(N)
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10, batch_size=32)

print(model.summary())
print(history.history.keys())


f = plt.figure(figsize=(10, 5))

# Optimisation history
sp =  f.add_subplot(1, 2, 1 )

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

sp =  f.add_subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig("training.png")
# plt.show()

# Generate prdiction case
# d_pred = []
# t_pred = []
# c_pred = []
# n_pred = 20
# for i in range(n_pred):
#     print(i)
#     t = random.uniform(onsagerIsingTc-1.5, onsagerIsingTc+1.5)
#     t_pred.append(t)
#     c_pred.append(0 if t < onsagerIsingTc else 1)
#     ising = IsingModel2D(N=N, T=t)
#     #print(ising)
#     ising.runMonteCarlo2(mcsteps)
#     d_pred.append(ising.lattice)
#     print(f"C = {c_pred[i]} ")
#     print(ising)

# t_pred = t_test
# c_pred = c_test
# d_pred = d_test
folder ="../data_100batches/"

d_pred = np.load(folder + "data.npy")
# energies = np.load(folder + "elist.npy")
t_pred = np.load(folder + "tlist.npy")
c_pred = np.load(folder + "clist.npy")
# magnetizations = np.load(folder +"mlist.npy")

f = plt.figure(figsize=(6, 5))

sp =  f.add_subplot(1, 1, 1 )
# pred = model(x_train[0:1])
pred = model(np.array(d_pred))
print(pred.numpy())


np.save('prediction-data', pred.numpy())
np.save('prediction-temp', t_pred)

plt.plot(t_pred,pred[:,0], '.')
plt.xlabel('Temperature')
plt.ylabel('Probability')
plt.grid(True)

plt.savefig("prediction.png")
#plt.show()
