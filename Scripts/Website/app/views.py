from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect

from os.path import join, exists, realpath, dirname
from os import mkdir, sep, remove
from glob import glob
from numpy import float32, array, argmax, zeros, uint8
from datetime import datetime
from cv2 import imread, imwrite, threshold, THRESH_BINARY, merge, bitwise_not, bitwise_and, bitwise_or, addWeighted, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from app.Utils.ImageClassification import resize as resizeClassification
from app.Utils.ImageSegmentation import resize as resizeSegmentation
from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout
from json import load

from os import mkdir, sep, environ

# Ocultar mensajes de Tensorflow
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# En caso de utilizar EfficientNet tiene clases y métricas personalizadas
class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape(inputs)[i] if sh is None else sh for i, sh in enumerate(self.noise_shape)])


customObjects = {
    'swish': nn.swish,
    'FixedDropout': FixedDropout
}
ClassNames = ['No humo', 'Humo']

# Lee el archivo de configuración
with open(join(dirname(realpath(__file__)), '..', 'Settings.json'), 'r') as f:
    data = load(f)
    Relative = join(str(data['Relative']))

# Crea los directorios necesarios
tempClassificationModelsDir = join(
    Relative, 'Models', 'TempClassificationModels')
if not exists(tempClassificationModelsDir):
    mkdir(tempClassificationModelsDir)

tempSegmentationModelsDir = join(Relative, 'Models', 'TempSegmentationModels')
if not exists(tempSegmentationModelsDir):
    mkdir(tempSegmentationModelsDir)

imagesDir = join(Relative, 'Images')
if not exists(imagesDir):
    mkdir(imagesDir)

# Variables globales
ModelName = ''

Model = None

CurrentIP = '127.0.0.1'

LastUsed = datetime.now()

clear_session()


# Página principal
def index(request):
    return render(request, 'index.html')


# Página de elección de predicción
def prediction(request):
    return render(request, 'prediccion.html')


# Página para la clasificación de imágenes
def classification(request):
    global CurrentIP
    global LastUsed
    global ModelName
    global Model
    ip = request.META['REMOTE_ADDR']
    if CurrentIP == ip:
        context = {}
        LastUsed = datetime.now()
        if request.method == 'POST':
            try:
                ModelName = request.POST['modelName']
                if request.POST.get('loadModel') and ModelName != '':
                    context['checkModel'] = 'model'
                    clear_session()
                    Model = load_model(
                        join(Relative, 'Models', ModelName), compile=False, custom_objects=customObjects)
                    prediction = Model.predict(
                        array([zeros((512, 512, 3), uint8)]).astype(float32), batch_size=1)
            except:
                print('')
            try:
                if request.POST.get('uploadModel'):
                    uploaded_file = request.FILES['model']
                    fs = FileSystemStorage()
                    name = fs.save(uploaded_file.name, uploaded_file)
                    with open(join(tempClassificationModelsDir, name), 'wb+') as destination:
                        for chunk in request.FILES['model'].chunks():
                            destination.write(chunk)
                if request.POST.get('uploadImage') and Model != None:
                    context['checkModel'] = 'model'
                    uploaded_file = request.FILES['image']
                    fs = FileSystemStorage()
                    name = fs.save(uploaded_file.name, uploaded_file)
                    context['url'] = fs.url(name)
                    prediction = Model.predict(array([resizeClassification([512, 512]).augment_image(cvtColor(
                        imread(join(imagesDir, name)), COLOR_BGR2RGB))/255.0]).astype(float32), batch_size=1)
                    predictClass = ClassNames[0 if prediction[0]
                                              [0] < 0.5 else 1]
                    context['prediction'] = predictClass
            except:
                print('No file upload')
        options = ''
        for file in list(glob(join(Relative, 'Models', 'Classification', '*'))) + list(glob(join(tempClassificationModelsDir, '*'))):
            if ModelName == sep.join(file.split(sep)[-2:]):
                options += '<option value=\"' + sep.join(file.split(sep)[-2:]) + '\" selected=\"selected\">' + (
                    file.split(sep)[-1]).split('.')[0] + '</option>'
            else:
                options += '<option value=\"' + \
                    sep.join(file.split(sep)[
                        -2:]) + '\">' + (file.split(sep)[-1]).split('.')[0] + '</option>'
        context['options'] = options
        return render(request, 'clasificacion.html', context)
    else:
        if (datetime.now() - LastUsed).total_seconds() > 300:
            CurrentIP = ip
            ModelName = ''
            clear_session()
            Model = None
            LastUsed = datetime.now()
            context = {}
            options = ''
            for f in glob(join(tempClassificationModelsDir, '*')):
                try:
                    remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
            for file in list(glob(join(Relative, 'Models', 'Classification', '*'))):
                if ModelName == sep.join(file.split(sep)[-2:]):
                    options += '<option value=\"' + sep.join(file.split(sep)[-2:]) + '\" selected=\"selected\">' + (
                        file.split(sep)[-1]).split('.')[0] + '</option>'
                else:
                    options += '<option value=\"' + \
                        sep.join(file.split(sep)[
                            -2:]) + '\">' + (file.split(sep)[-1]).split('.')[0] + '</option>'
            context['options'] = options
            return render(request, 'clasificacion.html', context)
        else:
            return render(request, 'prediccionOcupada.html')


# Página para la segmentación de imágenes
def segmentation(request):
    global CurrentIP
    global LastUsed
    global ModelName
    global Model
    ip = request.META['REMOTE_ADDR']
    if CurrentIP == ip:
        context = {}
        LastUsed = datetime.now()
        if request.method == 'POST':
            try:
                ModelName = request.POST['modelName']
                if request.POST.get('loadModel') and ModelName != '':
                    context['checkModel'] = 'model'
                    clear_session()
                    Model = load_model(
                        join(Relative, 'Models', ModelName), compile=False, custom_objects=customObjects)
                    prediction = Model.predict(
                        array([zeros((512, 512, 3), uint8)]).astype(float32), batch_size=1)
            except:
                print('')
            try:
                if request.POST.get('uploadModel'):
                    uploaded_file = request.FILES['model']
                    fs = FileSystemStorage()
                    name = fs.save(uploaded_file.name, uploaded_file)
                    with open(join(tempSegmentationModelsDir, name), 'wb+') as destination:
                        for chunk in request.FILES['model'].chunks():
                            destination.write(chunk)
                if request.POST.get('uploadImage') and Model != None:
                    context['checkModel'] = 'model'
                    uploaded_file = request.FILES['image']
                    fs = FileSystemStorage()
                    name = fs.save(uploaded_file.name, uploaded_file)
                    image = cvtColor(
                        imread(join(imagesDir, name)), COLOR_BGR2RGB)
                    (x, y, _) = image.shape
                    imageResized = resizeSegmentation(
                        [512, 512]).augment_image(image)
                    prediction = Model.predict(
                        array([imageResized/255.0]).astype(float32), batch_size=1)
                    thresh = threshold(
                        (prediction[0]*255.0).astype(uint8), 127, 255, THRESH_BINARY)[1].astype(uint8)
                    mask = bitwise_not(thresh)
                    blankImage = zeros((512, 512, 3), uint8)
                    blankImage[:, :] = (63, 255, 0)
                    imagePrepared = None
                    if x < y:
                        w = int(512*(1 - x/y)/2)
                        imagePrepared = cvtColor(addWeighted(imageResized, 0.5, bitwise_and(bitwise_or(merge([mask, mask, mask]), bitwise_and(
                            merge([thresh, thresh, thresh]), blankImage, None), None), imageResized, None), 0.5, 1), COLOR_RGB2BGR)[w:-w+1, :]
                    elif x > y:
                        w = int(512*(1 - y/x)/2)
                        imagePrepared = cvtColor(addWeighted(imageResized, 0.5, bitwise_and(bitwise_or(merge([mask, mask, mask]), bitwise_and(
                            merge([thresh, thresh, thresh]), blankImage, None), None), imageResized, None), 0.5, 1), COLOR_RGB2BGR)[:, w:-w+1]
                    else:
                        imagePrepared = cvtColor(addWeighted(imageResized, 0.5, bitwise_and(bitwise_or(merge([mask, mask, mask]), bitwise_and(
                            merge([thresh, thresh, thresh]), blankImage, None), None), imageResized, None), 0.5, 1), COLOR_RGB2BGR)
                    imwrite(join(imagesDir, 'TempImage.jpg'), imagePrepared)
                    context['url'] = fs.url('TempImage.jpg')
                    if sum(sum(thresh)) > 0:
                        context['leyenda'] = 'Humo'
                    else:
                        context['leyenda'] = 'No humo'
            except:
                print('No file upload')
        options = ''
        for file in list(glob(join(Relative, 'Models', 'Segmentation', '*'))) + list(glob(join(tempSegmentationModelsDir, '*'))):
            if ModelName == sep.join(file.split(sep)[-2:]):
                options += '<option value=\"' + sep.join(file.split(sep)[-2:]) + '\" selected=\"selected\">' + (
                    file.split(sep)[-1]).split('.')[0] + '</option>'
            else:
                options += '<option value=\"' + \
                    sep.join(file.split(sep)[
                        -2:]) + '\">' + (file.split(sep)[-1]).split('.')[0] + '</option>'
        context['options'] = options
        return render(request, 'segmentacion.html', context)
    else:
        if (datetime.now() - LastUsed).total_seconds() > 300:
            CurrentIP = ip
            ModelName = ''
            clear_session()
            Model = None
            LastUsed = datetime.now()
            context = {}
            options = ''
            for f in glob(join(tempSegmentationModelsDir, '*')):
                try:
                    remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
            for file in list(glob(join(Relative, 'Models', 'Segmentation', '*'))):
                if ModelName == sep.join(file.split(sep)[-2:]):
                    options += '<option value=\"' + sep.join(file.split(sep)[-2:]) + '\" selected=\"selected\">' + (
                        file.split(sep)[-1]).split('.')[0] + '</option>'
                else:
                    options += '<option value=\"' + \
                        sep.join(file.split(sep)[
                            -2:]) + '\">' + (file.split(sep)[-1]).split('.')[0] + '</option>'
            context['options'] = options
            return render(request, 'segmentacion.html', context)
        else:
            return render(request, 'prediccionOcupada.html')


# Página donde muestra la documentación
def documentation(request):
    return render(request, 'documentacion.html')
