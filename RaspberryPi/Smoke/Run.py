from datetime import datetime
from numpy import array
from imgaug.augmenters.size import CenterPadToAspectRatio, Resize
from imgaug.augmenters import Sequential
from cv2 import VideoCapture, cvtColor, threshold, imwrite
from tensorflow.keras.models import load_model
from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout
from sys import argv


class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape(inputs)[i] if sh is None else sh for i, sh in enumerate(self.noise_shape)])


# Carga del modelo
model = load_model('/root/Smoke/Models/model.h5', compile=False,
                   custom_objects={'swish': nn.swish, 'FixedDropout': FixedDropout})
listImages = []
# Captura la imagen en cada cámara
for i in range(1):
    listImages.append(VideoCapture(i).read()[1])
    VideoCapture(i).release()
for image in listImages:
    # Comprobar que la imagen contiene humo
    if sum(sum(threshold((model.predict(array([Sequential([CenterPadToAspectRatio(1., pad_mode='constant', pad_cval=0), Resize((512, 512), interpolation=3)]).augment_image(cvtColor(image, 4))/255.]).astype('float'), batch_size=1)[0][..., 0]*255.).astype('uint8'), 127, 255, 0)[1].astype('uint8'))) > 4096:
        # Send signal
        if argv[1] == 'true':
            # Guarda la imagen
            imwrite('/root/Smoke/Images/smoke/' +
                    datetime.now().strftime('%Y%m%d%H%M%S')+'.png', image)
    else:
        if argv[1] == 'true':
            # Guarda la imagen
            imwrite('/root/Smoke/Images/neutral/' +
                    datetime.now().strftime('%Y%m%d%H%M%S')+'.png', image)
