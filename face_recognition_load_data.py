from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from numpy import savez_compressed
from mtcnn.mtcnn import MTCNN
import Exceptions

# Funkcja ekstracji twarzy z obrazu
def extract_face(image, required_size=(160, 160), bounding_box=False):
    if Image.isImageType(image):
        pixels = asarray(image)
    else:
        pixels = image
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if results:
        # Wyznaczenie parametrów opisujących bounding box twarzy
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # Ekstrakcja rejonu obrazu, gdzie znajduje się twarz
        face = pixels[y1:y2, x1:x2]
        # Przeskalowanie obrazu twarzy
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        if bounding_box:
            return face_array, x1, x2, y1, y2
        else:
            return face_array
    else:
        raise Exceptions.NoFaceFoundError()

# Funkcja zapisująca zdjęcia danej klasy do jednego pliku (podkatalog ze zdjeciami oznacza konkretna klasę)
def load_dataset(directory):
    # Iterowanie po podkatalogach (klasach)
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        # Pomijanie plików w katalogu głównym, które nie się podkatalogami
        if not isdir(path):
            continue
        faces = list()
        counter = 0
        print('Loading faces for {}'.format(subdir))
        # Iteracja po obrazach znajdujących się danym podkatalogu
        for filename in listdir(path):
            img_path = path + filename
            image = Image.open(img_path)
            # konwersja obrazów do RGB
            image = image.convert('RGB')
            try:
                # Wywołanie funkcji ekstracji twarzy z obrazu
                face = extract_face(image)
                faces.append(face)
                counter = counter + 1
            except Exceptions.NoFaceFoundError:
                print('Could not find face at {}'.format(img_path))
        labels = [subdir for _ in range(len(faces))]
        # Zapis zdjeć wraz z nazwą odpowiadającem im klasy do pliku
        savez_compressed('celeb_faces/{}-faces.npz'.format(subdir), faces, labels)
        print('>Saved {} files for class {}'.format(len(faces), subdir))


# Wywołanie funkcji wczytującej zdjęcia na bazie zdjęc PINS Face recognition
load_dataset('105_classes_pins_dataset_only_color/')
