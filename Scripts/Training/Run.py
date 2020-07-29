import efficientnet.tfkeras as eftf
import Utils.Models as tfmo
import tensorflow.keras.applications as tfka

from segmentation_models import Unet
from segmentation_models.losses import categorical_focal_dice_loss, binary_focal_dice_loss
from segmentation_models.metrics import IOUScore
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import Utils.Callbacks as ca
from Utils.ImageTest import load as loadTest
from tensorflow.summary import create_file_writer, image
from os.path import exists, join, realpath, dirname
from os import mkdir
from numpy import array
from json import load as loadJSON
from shutil import copyfile

from Settings import optimizer, augmenters, currentPath, logDir, checkpointsDir, callbacks

# Lee el archivo de configuración
with open(join(dirname(realpath(__file__)), 'Settings.json'), 'r') as f:
    data = loadJSON(f)
    relativeData = join(str(data['RelativeData']))
    classNames = list(array(data['ClassNames']).astype(str))
    imageDimensions = list(array(data['ImageDimensions']).astype(int))
    mode = str(data['Mode'])
    modelName = str(data['ModelName'])
    weights = str(data['Weights'])
    decoderBlock = str(data['DecoderBlock'])
    batchSize = int(data['BatchSize'])
    epochs = int(data['Epochs'])

weights = None if weights == 'None' else weights

# Crea los directorios necesarios
if not exists(currentPath):
    mkdir(currentPath)
if not exists(logDir):
    mkdir(logDir)
if not exists(checkpointsDir):
    mkdir(checkpointsDir)

# Copia los archivos de configuración para futuras consultas
copyfile(join(dirname(realpath(__file__)), 'Settings.json'), join(currentPath, 'Settings.json'))
copyfile(join(dirname(realpath(__file__)), 'Settings.py'), join(currentPath, 'Settings.py'))

# Comprueba si el conjunto de datos será de clasificación o segmentación
if mode == 'classification':
    from Utils.ImageClassification import load, resize, DataGenerator, imageGrid, confusionMatrix, predict
else:
    from Utils.ImageSegmentation import load, resize, DataGenerator, imageGrid, confusionMatrix, predict

# Lee los archivos indicados
trainImages, trainLabels, validationImages, validationLabels = load(
    relativeData, classNames)

# Añade un ejemplo de imágenes al log de TensorBoard
fileWriter = create_file_writer(logDir)
fileWriter.set_as_default()
with fileWriter.as_default():
    image('Training data',
          ca.plotImages(imageGrid(trainImages,
                                  trainLabels,
                                  classNames=classNames,
                                  perRow=2,
                                  imageDimensions=(64, 64),
                                  nImages=8)),
          step=0)

with fileWriter.as_default():
    image('Validation data',
          ca.plotImages(imageGrid(validationImages,
                                  validationLabels,
                                  classNames=classNames,
                                  perRow=2,
                                  imageDimensions=(64, 64),
                                  nImages=8)),
          step=0)

# Configura las métricas según su modo
if mode == 'classification':
    if len(classNames) <= 2:
        activation = 'sigmoid'
        nClasses = 1

        loss = 'binary_crossentropy'
        metrics = 'binary_accuracy'
    else:
        trainLabels = to_categorical(trainLabels)
        validationLabels = to_categorical(validationLabels)

        activation = 'softmax'
        nClasses = len(classNames)

        loss = 'categorical_crossentropy'
        metrics = 'categorical_accuracy'
else:
    nClasses = len(classNames)

    metrics = IOUScore(threshold=0.5)

    if nClasses == 1:
        activation = 'sigmoid'
        loss = binary_focal_dice_loss
    else:
        activation = 'softmax'
        loss = categorical_focal_dice_loss

# Configura el generador de datos para usar data augmentation y redimensión
trainData = DataGenerator(trainImages,
                          trainLabels,
                          batchSize=batchSize,
                          resize=resize(imageDimensions[:-1]),
                          augmenters=augmenters,
                          shuffle=True)

validationData = DataGenerator(validationImages,
                               validationLabels,
                               batchSize=batchSize,
                               resize=resize(imageDimensions[:-1]))

# Según el modo se selecciona el modelo elegido o el backbone para U-Net
if mode == 'classification':
    try:
        modelFunction = getattr(tfmo, modelName)
    except:
        try:
            modelFunction = getattr(eftf, modelName)
        except:
            kerasApplications = [tfka.xception, tfka.vgg16, tfka.vgg19, tfka.resnet, tfka.resnet_v2, tfka.inception_v3,
                                 tfka.inception_resnet_v2, tfka.mobilenet, tfka.densenet, tfka.nasnet, tfka.mobilenet_v2]
            for ka in kerasApplications:
                try:
                    modelFunction = getattr(ka, modelName)
                    break
                except:
                    pass
    modelNoTop = modelFunction(input_shape=imageDimensions,
                               classes=nClasses,
                               weights=weights,
                               include_top=False,
                               pooling='avg')
    model = Sequential()
    model.add(modelNoTop)
    '''
    EfficientNet*   +Dropout
    VGG*            -AVG +Flatten +Dense +Dense
    MobileNet       +Reshape +Dropout +Conv2D +Reshape
    '''
    model.add(Dense(nClasses, activation=activation, name='predictions'))
else:
    model = Unet(backbone_name=modelName.lower(),
                 input_shape=imageDimensions,
                 classes=nClasses,
                 activation=activation,
                 encoder_weights=weights,
                 decoder_block_type=decoderBlock)
model.summary()

# Guarda el modelo antes de entrenar
model.save(join(currentPath, 'Model.h5'), include_optimizer=False)

# Compila el modelo con la configuración seleccionada
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[metrics])


# Añade al log una generación de matrix de confusión en cada época
def logConfusionMatrix(epoch, logs):
    with fileWriter.as_default():
        image('Confusion Matrix', ca.plotImages(confusionMatrix(
            model, validationImages, validationLabels, classNames, imageDimensions[:2])), step=epoch)


# Entrena el modelo
history = model.fit(trainData,
                    validation_data=validationData,
                    epochs=epochs,
                    callbacks=callbacks)

# Una vez terminado el entrenamiento carga y compila el modelo con los pesos del mejor checkpoint
bestModel = ca.bestCheckpointLoad(currentPath, checkpointsDir)
bestModel.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metrics])

# Imprime el resultado del mejor checkpoint
scores = bestModel.evaluate(validationData,
                            verbose=1)
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])

# Guarda el modelo con el mejor checkpoint en el directorio principal
bestModel.save(join(currentPath, 'ModelBestCheckpointAccuracy' +
                    str(scores[1])+'.h5'), include_optimizer=False)

# Carga las imágenes de prueba
testImage = loadTest(relativeData)

# Predice las imágenes de prueba
testLabels, totalTime, times = predict(
    currentPath, testImage, bestModel, classNames)

# Guarda la predición en TensorBoard
with fileWriter.as_default():
    image('Test:\nTotal time: '+str(totalTime)+'\nTime per image: '+str(sum(times)/len(times)),
          ca.plotImages(imageGrid(testImage,
                                  testLabels,
                                  classNames=classNames,
                                  perRow=1,
                                  imageDimensions=(64, 64),
                                  nImages=64)),
          step=0)
