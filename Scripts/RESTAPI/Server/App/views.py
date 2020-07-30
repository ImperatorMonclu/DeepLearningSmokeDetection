from os import mkdir
from App import app
import App.globals as gl

from os.path import join, exists, realpath, dirname
from random import randint
from datetime import datetime
from base64 import b64decode, b64encode
from json import load
from numpy import frombuffer, uint8, float32, array, argmax, zeros

from pydantic import BaseModel
from cv2 import imdecode, imwrite, imencode, threshold, THRESH_BINARY, merge, bitwise_not, bitwise_and, bitwise_or, addWeighted, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
from tensorflow.keras.models import load_model  # , model_from_json
from tensorflow.keras.backend import clear_session

from App.Utils.ImageClassification import resize as resizeClassification
from App.Utils.ImageSegmentation import resize as resizeSegmentation

from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout


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


# Lee el archivo de configuración
with open(join(dirname(realpath(__file__)), '..', 'Settings.json'), 'r') as f:
    data = load(f)
    Relative = join(str(data['Relative']))
    SegmentationColor = data['SegmentationColor']

modelsDir = join(Relative, 'Models')

# Crea los directorios necesarios
predictionsDir = join(Relative, 'Predictions')
if not exists(predictionsDir):
    mkdir(predictionsDir)


class ModelJSON(BaseModel):
    model: str = None
    classNames: list = None


class ImageJSON(BaseModel):
    image: str = None
    filename: str = None


# Recibe el nombre del modelo y lo carga
@app.post("/upload-model")
async def uploadModel(modelRequest: ModelJSON):
    clear_session()
    gl.Model = load_model(join(modelsDir, modelRequest.model+'.h5'),
                          compile=False, custom_objects=customObjects)
    # gl.Model = model_from_json(modelRequest.model)

    gl.Model.predict(array([zeros((int(gl.Model.input.shape[1]), int(
        gl.Model.input.shape[2]), int(gl.Model.input.shape[3])), dtype=float32)]), batch_size=1)

    modelDir = join(predictionsDir, modelRequest.model)
    if not exists(modelDir):
        mkdir(modelDir)
    gl.ModelName = modelRequest.model

    gl.ClassNames = modelRequest.classNames
    for cn in gl.ClassNames:
        classDir = join(modelDir, cn)
        if not exists(classDir):
            mkdir(classDir)


# Limpia la sesión de Tensorflow
@app.post("/clear-model")
async def clearModel():
    clear_session()


# Recibe la imagen serializada y manda predicción sobre clasificación
@app.post("/image-classification")
async def uploadImageClassification(imageRequest: ImageJSON):
    # Decodifica la imagen previamente serializada
    image = cvtColor(imdecode(frombuffer(
        b64decode((imageRequest.image).encode('ascii')), uint8), 1), COLOR_BGR2RGB)

    # Prepara la imagen con una correcta redimensión
    imageResized = resizeClassification([int(gl.Model.input.shape[1]), int(
        gl.Model.input.shape[2])]).augment_image(image)

    model = gl.Model
    imagePredicted = array([imageResized/255.0]).astype(float32)

    # Predice la imagen calculando el tiempo
    before = datetime.now()
    prediction = model.predict(imagePredicted, batch_size=1)
    after = datetime.now()

    # Comprueba que clase ha predicho
    if len(gl.ClassNames) > 2:
        predictClass = gl.ClassNames[argmax(prediction[0], axis=1)]
    else:
        predictClass = gl.ClassNames[0 if prediction[0][0] < 0.5 else 1]

    # Guarda la imagen original clasificada en el servidor
    classDir = join(predictionsDir, gl.ModelName, predictClass)
    imwrite(join(classDir, imageRequest.filename),
            cvtColor(image, COLOR_RGB2BGR))

    # Prepara la respuesta al cliente
    imageDict = imageRequest.dict()
    imageDict.update({"model": gl.ModelName})
    imageDict.update({"prediction": predictClass})
    imageDict.update({"time": (after - before).total_seconds()})

    return imageDict


# Recibe la imagen serializada y manda predicción sobre segmentación
@app.post("/image-segmentation")
async def uploadImageSegmentation(imageRequest: ImageJSON):
    # Decodifica la imagen previamente serializada
    image = cvtColor(imdecode(frombuffer(
        b64decode((imageRequest.image).encode('ascii')), uint8), 1), COLOR_BGR2RGB)
    (x, y, _) = image.shape

    # Prepara la imagen con una correcta redimensión
    imageResized = resizeSegmentation([int(gl.Model.input.shape[1]), int(
        gl.Model.input.shape[2])]).augment_image(image)

    model = gl.Model
    imagePredicted = array([imageResized/255.0]).astype(float32)

    # Predice la imagen calculando el tiempo
    before = datetime.now()
    prediction = model.predict(imagePredicted, batch_size=1)
    after = datetime.now()

    # Crea la máscara predicha por categoría
    listPrediction = []
    for i in range(prediction.shape[-1]):
        imagePrediction = (prediction[0][..., i]*255.0).astype(uint8)

        _, thresh = threshold(imagePrediction, 127, 255, THRESH_BINARY)
        thresh = thresh.astype(uint8)

        mask = bitwise_not(thresh)

        blankImage = zeros(
            (int(gl.Model.input.shape[1]), int(gl.Model.input.shape[2]), 3), uint8)
        blankImage[:, :] = SegmentationColor

        colortrans = addWeighted(imageResized, 0.5, bitwise_and(bitwise_or(merge([mask, mask, mask]), bitwise_and(
            merge([thresh, thresh, thresh]), blankImage, None), None), imageResized, None), 0.5, 1)

        imagePrepared = None
        if x < y:
            w = int(int(gl.Model.input.shape[1])*(1 - x/y)/2)
            thresh = thresh[w:-w+1, :]
            imagePrepared = colortrans[w:-w+1, :]
        elif x > y:
            w = int(int(gl.Model.input.shape[2])*(1 - y/x)/2)
            thresh = thresh[:, w:-w+1]
            imagePrepared = colortrans[:, w:-w+1]
        else:
            imagePrepared = colortrans

        _, data = imencode('.jpg', imagePrepared)
        _, dataThresh = imencode('.jpg', thresh)

        # Codifica las imágenes resultado junto a la máscara para mandarlo al cliente
        listPrediction.append(gl.ClassNames[i])
        listPrediction.append((b64encode(data.tobytes())).decode('ascii'))
        listPrediction.append((b64encode(dataThresh.tobytes())).decode('ascii'))

        # Guarda la imagen original y la máscara predicha
        modelDir = join(predictionsDir, gl.ModelName)
        imageDir = join(modelDir, 'Images')
        if not exists(imageDir):
            mkdir(imageDir)
        imwrite(join(imageDir, (imageRequest.filename).split(
            '.')[0]+'.png'), cvtColor(image, COLOR_RGB2BGR))
        imwrite(join(modelDir, gl.ClassNames[i], (imageRequest.filename).split(
            '.')[0]+'.png'), thresh)

    # Prepara la respuesta al cliente
    imageDict = imageRequest.dict()
    imageDict.update({"model": gl.ModelName})
    imageDict.update({"prediction": listPrediction})
    imageDict.update({"time": (after - before).total_seconds()})

    return imageDict


# URL principal
@app.get("/")
async def index():
    return {"msg": "Hello World"}
