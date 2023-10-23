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
folder = "../data/"

meta = np.load(folder + "meta.npy")
data = np.load(folder + "data.npy")
energies = np.load(folder + "elist.npy")
temperatures = np.load(folder + "tlist.npy")
categories = np.load(folder + "clist.npy")
magnetizations = np.load(folder + "mlist.npy")


N = int(meta[0])
mcsteps = int(meta[1])
t_range = meta[2]

print(f"Data size {data.shape}")
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

plt.savefig("T-isingdata.png")

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
print("Compiling and training ML algorithm.")

x_train, y_train = np.array(d_train), np.array(t_train)
x_test, y_test = np.array(d_test), np.array(t_test)

def conv1(N):
  model = keras.Sequential()

  model.add(Conv2D(filters = 64, kernel_size = (2,2),padding = 'Same',
                  activation ='relu', input_shape = (N,N,1)))
  model.add(Flatten())
  model.add(Dense(64, activation = "sigmoid"))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  optimizer = keras.optimizers.Adam()

  model.compile(optimizer=optimizer,
                loss=keras.losses.MeanSquaredError()
                ,metrics=['mean_squared_error']
                )

  return model

def conv2(N):
  """
  mean err = 0.15
  data set = data10000_N16x16_rand
  """
  model = keras.Sequential()
  model.add(keras.Input(shape = (N,N,1)))
  model.add(Conv2D(filters = N*N/4, kernel_size = (2,2),padding = 'Same',
                  activation ='relu'))
  model.add(MaxPool2D(pool_size=(2,2)))
  model.add(Conv2D(filters = N, kernel_size = (4,4),padding = 'Same',
                  activation ='relu'))
  model.add(Flatten())
  model.add(Dense(N*N/4, activation = "relu"))
  model.add(Dropout(0.6))
  model.add(Dense(1))
  optimizer = keras.optimizers.Adam()

  model.compile(optimizer=optimizer,
                loss=keras.losses.MeanSquaredError()
                ,metrics=['mean_squared_error']
                )
  print(model.summary())
  return model


def nature1(N):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(N,N)),
        keras.layers.Dense(100, activation='sigmoid'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    return model

def nature2(N):
  model = keras.Sequential()

  model.add(Conv2D(filters = 64, kernel_size = (2,2),padding = 'Same',
                  activation ='relu', input_shape = (N,N,1)))

  model.add(Flatten())
  model.add(Dense(64, activation = "relu"))
  model.add(Dropout(0.6))
  model.add(Dense(2, activation='softmax'))
  optimizer = keras.optimizers.Adam()

  model.compile(optimizer=optimizer,
                loss=keras.losses.MeanSquaredError(),
                metrics=['accuracy'])

  return model



case = "-conv2"
model = conv2(N)

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=14, batch_size=32)


# f = plt.figure(figsize=(18, 10))
print(history.history)
print(model.summary())
print(history.history.keys())

f = plt.figure(figsize=(10, 5))

# Optimisation history
# sp =  f.add_subplot(1, 2, 1 )
#
# plt.plot(history.history['mean_squared_error'], label='Training mean_squared_error')
# plt.plot(history.history['val_mean_squared_error'], label='Validation mean_squared_error')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)

sp =  f.add_subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig("T-training" + case + ".png")

# Generate prdiction case
# d_pred = []
# t_pred = []
# # c_pred = []
# n_pred = 10
# t = random.uniform(onsagerIsingTc-t_range/2, onsagerIsingTc+t_range/2)
# useSameConfig = False
#
# if useSameConfig:
#     ising = IsingModel2D(N=N, T=t)
#     print(ising)
#     ising.runMonteCarlo2(mcsteps)
#
# for i in range(n_pred):
#     print(i)
#     # t = random.uniform(onsagerIsingTc-1.5, onsagerIsingTc+1.5)
#     if not useSameConfig:
#         ising = IsingModel2D(N=N, T=t)
#         ising.runMonteCarlo2(mcsteps)
#         print(ising)
#
#     t_pred.append(t)
#     d_pred.append(ising.lattice)

# t_pred = t_test
# c_pred = c_test
# d_pred = d_test
folder ="../data100/"
d_pred = np.load(folder + "data.npy")
t_pred = np.load(folder + "tlist.npy")

pred = model(np.array(d_pred))

f = plt.figure(figsize=(18, 10))
sp =  f.add_subplot(1, 3, 1 )

print(np.array(t_pred))
np.save('T-prediction-data' + case , pred.numpy())
np.save('T-prediction-temp' + case , t_pred)

print(pred.numpy().flatten())
plt.plot(pred, 'r.', label='Prediction')
plt.plot(t_pred, 'b.', label='Actual')
plt.xlabel('Cases')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)

sp =  f.add_subplot(1, 3, 2 )

diff = np.abs(np.array(t_pred)-pred.numpy().flatten())
# print(diff.shape)
plt.plot(t_pred, diff, 'r.', label='Difference')
plt.xlabel('Cases')
plt.ylabel('Temperature')
plt.title(f"Mean diff={np.average(diff)}")
plt.legend()
plt.grid(True)

sp =  f.add_subplot(1, 3, 3)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig("T-prediction" + case +".png")
# plt.show()
