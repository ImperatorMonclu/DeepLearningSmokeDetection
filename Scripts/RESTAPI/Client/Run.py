from os import mkdir, sep

from os.path import join, exists, realpath, dirname
from glob import glob
from base64 import b64encode, b64decode
from requests import post
from json import dumps, loads, load
from numpy import frombuffer, uint8, array

from cv2 import imread, imencode, imdecode, imwrite, cvtColor, COLOR_RGB2BGR

# Pendiente para pasar modelos por clientes
'''
from tensorflow.keras.backend import clear_session #, shape
from tensorflow.keras.models import load_model
'''
'''
from tensorflow import nn
from tensorflow.keras.layers import Dropout


class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape(inputs)[i] if sh is None else sh for i, sh in enumerate(self.noise_shape)])


customObjects = {
    'swish': nn.swish,
    'FixedDropout': FixedDropout
}
'''

# Lee el archivo de configuración
with open(join(dirname(realpath(__file__)), 'Settings.json'), 'r') as f:
    data = load(f)
    url = str(data['Host']) + ':' + str(int(data['Port']))
    relative = join(str(data['Relative']))
    modelClassificationNames = data['Classification']['ModelNames']
    classClassificationNames = data['Classification']['ClassNames']
    modelSegmentationNames = data['Segmentation']['ModelNames']
    classSegmentationNames = data['Segmentation']['ClassNames']


#modelsDir = join(relative, 'Models')


# Manda el nombre del modelo para cargarlo
def uploadModel(model, classNames):
    '''
    clear_session()
    model_json = load_model(
        join(modelsDir, model), compile=false, custom_objects=customObjects).to_json()
    '''
    response = post(url + '/upload-model',
                    data=dumps({'model': model, 'classNames': classNames}))
    return response


# Limpia el modelo cargado
def clearModel():
    response = post(url + '/clear-model')
    return response


# Creación de directorios
imagesDir = join(relative, 'Images')

predictionsDir = join(relative, 'Predictions')
if not exists(predictionsDir):
    mkdir(predictionsDir)

classificationDir = join(predictionsDir, 'Classification')
if not exists(classificationDir):
    mkdir(classificationDir)


# Manda las imágenes para clasificar
def imageClassification(filename):
    _, data = imencode('.jpg', imread(join(imagesDir, filename), 1))
    response = post(url+'/image-classification', data=dumps({'image': (b64encode(
        data.tobytes())).decode('ascii'), 'filename': filename.split(sep)[-1]}))

    reponseJSON = loads(response.content)

    image = imdecode(frombuffer(
        b64decode(reponseJSON['image'].encode('ascii')), uint8), 1)

    modelDir = join(classificationDir, reponseJSON['model'])
    if not exists(modelDir):
        mkdir(modelDir)

    classDir = join(modelDir, reponseJSON['prediction'])
    if not exists(classDir):
        mkdir(classDir)

    imwrite(join(classDir, reponseJSON['filename']), image)

    return reponseJSON['time']


segmentationDir = join(predictionsDir, 'Segmentation')
if not exists(segmentationDir):
    mkdir(segmentationDir)


# Manda las imágenes para segmentar
def imageSegmentation(filename):
    _, data = imencode('.png', imread(join(imagesDir, filename), 1))
    response = post(url+'/image-segmentation', data=dumps({'image': (b64encode(
        data.tobytes())).decode('ascii'), 'filename': filename.split(sep)[-1]}))

    reponseJSON = loads(response.content)

    modelDir = join(segmentationDir, reponseJSON['model'])
    if not exists(modelDir):
        mkdir(modelDir)

    for i in range(len(reponseJSON['prediction'])):
        if i % 3 == 0:
            classDir = join(modelDir, reponseJSON['prediction'][i])
            if not exists(classDir):
                mkdir(classDir)
            maskDir = join(classDir, 'Mask')
            if not exists(maskDir):
                mkdir(maskDir)
        elif i % 3 == 1:
            imwrite(join(classDir, reponseJSON['filename'].split('.')[0]+'.png'), cvtColor(imdecode(
                frombuffer(b64decode(reponseJSON['prediction'][i].encode('ascii')), uint8), 1), COLOR_RGB2BGR))
        else:
            imwrite(join(maskDir, reponseJSON['filename'].split('.')[0]+'.png'), imdecode(
                frombuffer(b64decode(reponseJSON['prediction'][i].encode('ascii')), uint8), 1))

    return reponseJSON['time']


# Lanza las peticiones de clasificación de imágenes
for i in range(len(modelClassificationNames)):
    print(uploadModel(str(modelClassificationNames[i]), list(
        array(classClassificationNames[i]).astype(str))))
    filenames = list(glob(join(imagesDir, '*')))
    times = []
    for fn in filenames:
        times.append(imageClassification(fn))
    print(sum(times)/len(times))

# Lanza las peticiones de segmentación de imágenes
for i in range(len(modelSegmentationNames)):
    print(uploadModel(str(modelSegmentationNames[i]), list(
        array(classSegmentationNames[i]).astype(str))))
    filenames = list(glob(join(imagesDir, '*')))
    times = []
    for fn in filenames:
        times.append(imageSegmentation(fn))
    print(sum(times)/len(times))

# Manda limpiar la ejecución de Tensorflow
print(clearModel())
