import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras_vggface.vggface import VGGFace
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Input
from keras.engine import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD


# Wczytanie bazy zdjęć FER2013
fer_data = np.load('FER2013-faces-without-disgust.npz')
fer_faces, fer_labels = fer_data['arr_0'], fer_data['arr_1']
print('Loaded: ', fer_faces.shape, fer_labels.shape)
print(len(set(fer_labels)))

# Wczytanie bazy zdjęć CK+
ck_data = np.load('CK2_without_contempt_disgust_3frames_faces.npz')
ck_faces, ck_labels = ck_data['arr_0'], ck_data['arr_1']
print('Loaded: ', ck_faces.shape, ck_labels.shape)
print(len(set(ck_labels)))

# Zmiana nazw klas bazy CK+, aby były takie same jak w przypadku bazy FER2013
dict = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'sad', 4: 'surprise'}
new_ck_labels = []
for label in ck_labels:
    new_ck_labels.append(dict[label])
new_ck_labels = np.asarray(new_ck_labels)

# Połączenie bazy FER2013 i CK+ w jedną bazę
faces = np.concatenate((fer_faces, ck_faces))
labels = np.concatenate((fer_labels, new_ck_labels))
print('Final shape: ', faces.shape, labels.shape)

plt.hist(labels)
plt.show()

# Zakodowanie klas do postaci liczb naturalnych
out_encoder = LabelEncoder()
out_encoder.fit(labels)
labels = out_encoder.transform(labels)
# Zakodowanie klas do postaci binarnej
categorical_labels = to_categorical(labels, num_classes=5)

# Podział danych na zbiór treningowy, walidacyjny i testowy
seed = 7
x_train, x_validation, y_train, y_validation = train_test_split(faces, categorical_labels, test_size=0.3, random_state=seed, stratify=categorical_labels)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation, y_validation, test_size=0.5, random_state=seed, stratify=y_validation)

# Każde zdjęcie treningowe zostanie poddane normalizacji (przeskalowanie pikseli) oraz poddane losowym transformacją (rotacja, odbicie horyzontalne)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=2,
    horizontal_flip=True
)

# Każde zdjęcie testowe zostanie poddane przeskalowaniu
test_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values to [0,1]
)

# Stworzenie generatora zdjęć treningowych
train_generator = train_datagen.flow(
    x=x_train,
    y=y_train,
    batch_size=24
)

# Stworzenie generatora zdjęć walidacyjnych
validation_generator = test_datagen.flow(
    x=x_validation,
    y=y_validation,
    batch_size=4
)

# Wczytanie modelu bazowego
base_model = VGGFace(include_top=False, input_shape=(160, 160, 3))
# Zablokowanie nauki warstw modelu bazowego
base_model.trainable = False

# Stworzenie sieci neuronowej
model = Sequential([
    # Model bazowy
    base_model,
    # Funkcja flatten
    Flatten(name='flatten'),
    # Warstwa wyjściowa o funkcji aktywacji softmax
    Dense(5, activation='softmax', name='classifier')
])

# Zasady nauki sieci neuronowej
model.compile(loss='categorical_crossentropy',  # Funkcja strat
              optimizer="adam",  # Optymalizator (learning rate = 0.01)
              metrics=['accuracy'])
# Wyświetlenie struktury stworzonej sieci
model.summary()

# Zapisywanie wag sieci neuronowej w trakcie uczenia, jeśli dochodzi do zwiększenia dokładności predykcji modelu
filepath = "FER2013_CK2_model_VGGFace_validation_and_testing_weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Rozpoczęcie nauki sieci
history = model.fit_generator(
    generator=train_generator,
    epochs=10,  # Liczba epok nauki
    validation_data=validation_generator,
    callbacks=callbacks_list)

# Wyświetlenie dokładności predykcji modelu na zbiorze treningowym i walidacyjnym w zależności od epoki nauki
plt.figure()
plt.plot(history.history['accuracy'], 'orange', label='Training accuracy')
plt.plot(history.history['val_accuracy'], 'blue', label='Validation accuracy')
plt.legend()
plt.show()
# Wyświetlenie wyniku funkcji strat dla zbioru treningowego i walidacyjnego w zależności od epoki nauki
plt.figure()
plt.plot(history.history['loss'], 'red', label='Training loss')
plt.plot(history.history['val_loss'], 'green', label='Validation loss')
plt.legend()
plt.show()

# Zapis nauczonego modelu do pliku
model.save('FER2013_CK2_model_VGGFace__validation_and_testing.h5')

