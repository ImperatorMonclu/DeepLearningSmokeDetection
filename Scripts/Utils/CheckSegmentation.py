from os import mkdir, sep
from os.path import join, exists
from datetime import datetime
from glob import glob
from json import load
from numpy import uint8, float32, array, argmax, zeros
from numpy.random import shuffle

from cv2 import imread, imwrite, threshold, THRESH_BINARY, merge, bitwise_not, bitwise_and, bitwise_or, addWeighted, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
from tensorflow.keras.models import load_model

from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout

from imgaug.augmenters import Sequential
from imgaug.augmenters.size import CenterPadToAspectRatio, Resize
from cv2 import INTER_AREA


class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape(inputs)[i] if sh is None else sh for i, sh in enumerate(self.noise_shape)])


customObjects = {
    'swish': nn.swish,
    'FixedDropout': FixedDropout
}


def resize(imageDimensions):
    return Sequential(
        [
            CenterPadToAspectRatio(float(imageDimensions[0])/float(imageDimensions[1]),
                                   pad_mode='constant',
                                   pad_cval=0),
            Resize(imageDimensions,
                   interpolation=INTER_AREA)
        ])


relative = "C:\\Users\\mauro\\Documents\\Minsait\\NoOneDrive\\Humos\\Prediction"
relativeData = "C:\\Users\\mauro\\Documents\\Minsait\\NoOneDrive\\Humos\\Data\\Classification"

modelName = 'UNetEfficientNetB4Tranpose'
classNames = ['smoke']

# ******************************
Value = 0
# ******************************

predictionsIndraDir = join(relative, 'PredictionsIndra')
if not exists(predictionsIndraDir):
    mkdir(predictionsIndraDir)
for cn in classNames:
    classDir = join(predictionsIndraDir, cn)
    if not exists(classDir):
        mkdir(classDir)


trainFilenamesSmoke = list(glob(join(relativeData, 'Train', 'smoke', '*')))
trainFilenamesNeutral = list(glob(join(relativeData, 'Train', 'neutral', '*')))
validationFilenamesSmoke = list(
    glob(join(relativeData, 'Validation', 'smoke', '*')))
validationFilenamesNeutral = list(
    glob(join(relativeData, 'Validation', 'neutral', '*')))
filenamesSmoke = trainFilenamesSmoke + validationFilenamesSmoke
filenamesNeutral = trainFilenamesNeutral + validationFilenamesNeutral


model = load_model(join(relative, 'Models', modelName+'.h5'),
                   compile=False, custom_objects=customObjects)

n = 0
a = 0
smokeList = []
for filename in filenamesSmoke:
    n += 1

    image = cvtColor(imread(filename), COLOR_BGR2RGB)
    imageResized = resize([int(model.input.shape[1]), int(
        model.input.shape[2])]).augment_image(image)
    imagePredicted = array([imageResized/255.0]).astype(float32)

    prediction = model.predict(imagePredicted, batch_size=1)

    imagePrediction = (prediction[0]*255.0).astype(uint8)
    _, thresh = threshold(imagePrediction, 127, 255, THRESH_BINARY)
    thresh = thresh.astype(uint8)
    nThresh = sum(sum(thresh))
    smokeList.append(nThresh)

    if nThresh > Value:
        a += 1

    print((n*100)/(len(filenamesNeutral)+len(filenamesSmoke)))

aSmoke = a

neutralList = []
for filename in filenamesNeutral:
    n += 1

    image = cvtColor(imread(filename), COLOR_BGR2RGB)
    imageResized = resize([int(model.input.shape[1]), int(
        model.input.shape[2])]).augment_image(image)
    imagePredicted = array([imageResized/255.0]).astype(float32)

    prediction = model.predict(imagePredicted, batch_size=1)

    imagePrediction = (prediction[0]*255.0).astype(uint8)
    _, thresh = threshold(imagePrediction, 127, 255, THRESH_BINARY)
    thresh = thresh.astype(uint8)
    nThresh = sum(sum(thresh))
    neutralList.append(nThresh)

    if nThresh <= Value:
        a += 1

    print((n*100)/(len(filenamesNeutral)+len(filenamesSmoke)))

aNeutral = a - aSmoke


def median(lst):
    n = len(lst)
    s = sorted(lst)
    return (sum(s[n//2-1:n//2+1])/2.0, s[n//2])[n % 2] if n else None


print('Maximum neutral '+str(max(neutralList)))
print('Minimum smoke '+str(min(smokeList)))
print('Average neutral '+str(sum(neutralList)/len(neutralList)))
print('Average smoke '+str(sum(smokeList)/len(smokeList)))
print('Median neutral '+str(median(neutralList)))
print('Median smoke '+str(median(smokeList)))
print('Percentage neutral '+str(aNeutral/len(filenamesNeutral)))
print('Percentage smoke '+str(aSmoke/len(filenamesSmoke)))
print('Percentage '+str(a/n))
