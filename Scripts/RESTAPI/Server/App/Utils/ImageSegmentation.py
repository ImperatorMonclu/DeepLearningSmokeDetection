from imgaug.augmenters import Sequential
from imgaug.augmenters.size import CenterPadToAspectRatio, Resize
from cv2 import INTER_AREA


# Redimensiona la imagen
def resize(imageDimensions):
    return Sequential(
        [
            CenterPadToAspectRatio(float(imageDimensions[0])/float(imageDimensions[1]),
                                   pad_mode='constant',
                                   pad_cval=0),
            Resize(imageDimensions,
                   interpolation=INTER_AREA)
        ])
