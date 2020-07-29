from datetime import datetime
beforeAll = datetime.now()

from tensorflow.keras.layers import Dropout
from tensorflow.keras.backend import shape
from tensorflow import nn
from tensorflow.keras.models import load_model
from cv2 import imread, cvtColor, threshold, imwrite, VideoCapture
from imgaug.augmenters import Sequential
from imgaug.augmenters.size import CenterPadToAspectRatio, Resize
from numpy import array


class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape(inputs)[i] if sh is None else sh for i, sh in enumerate(self.noise_shape)])


# Carga del modelo
model = load_model('/root/Smoke/Models/model.h5', compile=False,
                   custom_objects={'swish': nn.swish, 'FixedDropout': FixedDropout})
# Creación de la imagen con humo redimensionada
imageSmokeAug = Sequential([CenterPadToAspectRatio(1., pad_mode='constant', pad_cval=0), Resize(
    (512, 512), interpolation=3)]).augment_image(cvtColor(imread('/root/Smoke/Test/smoke.png'), 4))
imageSmoke = array([imageSmokeAug/255.]).astype('float')
# Comprueba el tiempo que tarda en predecir la imagen con humo (Siempre tardará más en la primera predicción)
before = datetime.now()
# Predice la imagen con humo
predictionSmoke = model.predict(imageSmoke, batch_size=1)
after = datetime.now()
print('Prediction time smoke image: '+str((after - before).total_seconds()))
# Creación de la imagen sin humo redimensionada
imageNeutralAug = Sequential([CenterPadToAspectRatio(1., pad_mode='constant', pad_cval=0), Resize(
    (512, 512), interpolation=3)]).augment_image(cvtColor(imread('/root/Smoke/Test/neutral.png'), 4))
imageNeutral = array([imageNeutralAug/255.]).astype('float')
# Comprueba el tiempo que tarda en predecir la imagen sin humo
before = datetime.now()
# Predice la imagen sin humo
predictionNeutral = model.predict(imageNeutral, batch_size=1)
after = datetime.now()
print('Prediction time neutral image: '+str((after - before).total_seconds()))
# Crea las máscaras de cada imagen
maskSmoke = threshold(
    (predictionSmoke[0][..., 0]*255.).astype('uint8'), 127, 255, 0)[1].astype('uint8')
maskNeutral = threshold(
    (predictionNeutral[0][..., 0]*255.).astype('uint8'), 127, 255, 0)[1].astype('uint8')
# Guarda las imágenes redimensionadas y las máscaras predecidas
imwrite('/root/Smoke/Test/smokeMask.png', maskSmoke)
imwrite('/root/Smoke/Test/smokeAug.png', cvtColor(imageSmokeAug, 4))
imwrite('/root/Smoke/Test/neutralMask.png', maskNeutral)
imwrite('/root/Smoke/Test/neutralAug.png', cvtColor(imageNeutralAug, 4))
# Imprime si la predicción ha fallado o ha sido exitosa
if sum(sum(maskSmoke)) <= 0 or sum(sum(maskNeutral)) > 0:
    print('Prediction FAILED!')
else:
    print('Prediction SUCCESSFUL!')
# Realiza una captura de cada cámara
for i in range(1):
    imwrite('/root/Smoke/TestCam/'+'imageCamera' +
            str(i)+'.png', VideoCapture(i).read()[1])
    VideoCapture(i).release()
# Devuelve el tiempo total que le ha llevado la prueba
afterAll = datetime.now()
print('Time: '+str((afterAll - beforeAll).total_seconds()))
