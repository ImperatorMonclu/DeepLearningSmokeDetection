from tensorflow.keras.models import load_model

from tensorflow.summary import create_file_writer, image
from os.path import join, realpath, dirname
from os import sep, mkdir
from glob import glob
from numpy import array, argmax, float32, uint8, zeros
from numpy.random import shuffle
from json import load as loadJSON

from glob import glob
from os.path import join, exists

from imgaug.augmenters import Sequential
from imgaug.augmenters.size import CenterPadToAspectRatio, Resize
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from cv2 import INTER_AREA, imread, cvtColor, COLOR_BGR2RGB, imwrite, COLOR_RGB2BGR, IMREAD_UNCHANGED, threshold, THRESH_BINARY, merge, bitwise_not, bitwise_and, bitwise_or, addWeighted

import matplotlib.pyplot as plt
from datetime import datetime

from io import BytesIO
from tensorflow import expand_dims, image as _image

from tensorflow import nn
from tensorflow.keras.backend import shape
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


with open(join(dirname(realpath(__file__)), 'Settings.json'), 'r') as f:
    data = loadJSON(f)
    currentPath = join(str(data['CurrentPath']))
    relativeData = join(str(data['RelativeData']))
    classNames = list(array(data['ClassNames']).astype(str))
    imageDimensions = list(array(data['ImageDimensions']).astype(int))
    mode = str(data['Mode'])

checkpointsDir = join(currentPath, 'Checkpoints')
logDir = join(currentPath, 'Log')


if mode == 'classification':
    def resize(imageDimensions):
        return Sequential(
            [
                CenterPadToAspectRatio(float(imageDimensions[0])/float(imageDimensions[1]),
                                       pad_mode='edge'),
                Resize(imageDimensions,
                       interpolation=INTER_AREA)
            ])

    def imageGrid(images, labels, classNames=['0'], perRow=4, imageDimensions=(64, 64), nImages=16):
        rows = int(nImages/perRow)
        figure, axarr = plt.subplots(rows, perRow, figsize=(
            imageDimensions[0], int(imageDimensions[1]*rows/perRow)))
        k = 0
        for i in range(rows):
            if perRow > 1:
                for j in range(perRow):
                    axarr[i, j].imshow(resize(imageDimensions).augment_image(
                        cvtColor(imread(images[k][0]), COLOR_BGR2RGB)))
                    axarr[i, j].set_title(
                        classNames[1 if labels[k][0] > 0.5 else 0])
                    axarr[i, j].axis(False)
                    k = + 1
            else:
                axarr[i].imshow(resize(imageDimensions).augment_image(
                    cvtColor(imread(images[k][0]), COLOR_BGR2RGB)))
                axarr[i].set_title(classNames[1 if labels[k][0] > 0.5 else 0])
                axarr[i].axis(False)
                k = + 1
        plt.subplots_adjust(wspace=0, hspace=0.125,
                            left=0, right=1, bottom=0, top=1)
        return figure

    def predict(relativePath, testData, model, classNames):
        beforeAll = datetime.now()

        predictionsDir = join(relativePath, 'Predictions')
        if not exists(predictionsDir):
            mkdir(predictionsDir)

        for cn in classNames:
            classDir = join(predictionsDir, cn)
            if not exists(classDir):
                mkdir(classDir)

        times = []
        listPredictions = []
        for i in range(len(testData)):
            image = cvtColor(imread(testData[i][0]), COLOR_BGR2RGB)

            imageResized = resize([int(model.input.shape[1]), int(
                model.input.shape[2])]).augment_image(image)

            imagePredicted = array([imageResized/255.0]).astype(float32)

            before = datetime.now()
            prediction = model.predict(imagePredicted, batch_size=1)
            after = datetime.now()
            times.append((after - before).total_seconds())

            if len(classNames) > 2:
                imwrite(join(predictionsDir, classNames[argmax(
                    prediction[0], axis=1)], testData[i][0].split(sep)[-1]), cvtColor(image, COLOR_RGB2BGR))
            else:
                imwrite(join(predictionsDir, classNames[1 if prediction[0][0] > 0.5 else 0], testData[i][0].split(
                    sep)[-1]), cvtColor(image, COLOR_RGB2BGR))

            listPredictions.append(prediction[0])

        afterAll = datetime.now()

        return array(listPredictions), (afterAll - beforeAll).total_seconds(), times
else:
    def resize(imageDimensions):
        return Sequential(
            [
                CenterPadToAspectRatio(float(imageDimensions[0])/float(imageDimensions[1]),
                                       pad_mode='constant',
                                       pad_cval=0),
                Resize(imageDimensions,
                       interpolation=INTER_AREA)
            ])

    def imageGrid(images, labels, classNames=['0'], perRow=2, imageDimensions=(64, 64), nImages=16):
        nClasses = len(classNames)+1
        resizeImage = resize(imageDimensions)
        columns = perRow*nClasses
        rows = int((nImages*nClasses)/columns)
        figure, axarr = plt.subplots(rows, columns, figsize=(
            imageDimensions[0], int(imageDimensions[1]*rows/columns)))
        k = 0
        for i in range(rows):
            for j in range(columns):
                if j % nClasses == 0:
                    k = int(j/nClasses)+(i*perRow)
                    axarr[i, j].imshow(resizeImage.augment_image(
                        cvtColor(imread(images[k][0]), COLOR_BGR2RGB)))
                    axarr[i, j].set_title(
                        (images[k][0].split(sep)[-1]).split('.')[0])
                else:
                    mask = imread(labels[j % nClasses-1]
                                  [k][0], IMREAD_UNCHANGED)
                    axarr[i, j].imshow(resizeImage.augment_segmentation_maps(
                        [SegmentationMapsOnImage(mask, shape=mask.shape)])[0].get_arr())
                    axarr[i, j].set_title(classNames[j % nClasses-1])
                axarr[i, j].axis(False)
        plt.subplots_adjust(wspace=0, hspace=0.0675,
                            left=0, right=1, bottom=0, top=1)
        return figure

    def predict(relativePath, testData, model, classNames):
        beforeAll = datetime.now()

        predictionsDir = join(relativePath, 'Predictions')
        if not exists(predictionsDir):
            mkdir(predictionsDir)

        for cn in classNames:
            classDir = join(predictionsDir, cn)
            if not exists(classDir):
                mkdir(classDir)

        times = []
        listPredictions = []
        for i in range(len(testData)):
            image = cvtColor(imread(testData[i][0]), COLOR_BGR2RGB)

            imageResized = resize([int(model.input.shape[1]), int(
                model.input.shape[2])]).augment_image(image)

            imagePredicted = array([imageResized/255.0]).astype(float32)

            before = datetime.now()
            prediction = model.predict(imagePredicted, batch_size=1)
            after = datetime.now()
            times.append((after - before).total_seconds())

            listClassPredictions = []
            for c in range(prediction.shape[-1]):
                imagePrediction = (prediction[0][..., c]*255.0).astype(uint8)

                _, thresh = threshold(imagePrediction, 127, 255, THRESH_BINARY)
                thresh = thresh.astype(uint8)

                mask = bitwise_not(thresh)

                blankImage = zeros(
                    (int(model.input.shape[1]), int(model.input.shape[2]), 3), uint8)
                blankImage[:, :] = (63, 255, 0)

                colortrans = addWeighted(imageResized, 0.5, bitwise_and(bitwise_or(merge([mask, mask, mask]), bitwise_and(
                    merge([thresh, thresh, thresh]), blankImage, None), None), imageResized, None), 0.5, 1)

                fileName = join(
                    predictionsDir, classNames[c], (testData[i][0].split(sep)[-1]).split('.')[0])
                imwrite(fileName+'.', cvtColor(colortrans, COLOR_RGB2BGR))
                imwrite(fileName+'.png', thresh)

                listClassPredictions.append(fileName+'.png')

            listPredictions.append(array(listClassPredictions).reshape(
                (len(listClassPredictions), 1)))

        afterAll = datetime.now()

        return array(listPredictions).transpose(1, 0, 2), (afterAll - beforeAll).total_seconds(), times


checkpointFilenames = list(glob(join(checkpointsDir, '*')))
xMax = 0.0
for f in checkpointFilenames:
    x = float(f.split(sep)[-1][-9:-3])
    if x > xMax:
        xMax = x
        checkpoint = f
bestModel = load_model(join(currentPath, 'Model.h5'), customObjects)
bestModel.load_weights(checkpoint)

bestModel.save(join(currentPath, 'ModelAccuracy' + str(xMax)+'.h5'),
               include_optimizer=False)

testImage = list(glob(join(relativeData, 'Test', '*')))
shuffle(testImage)
testImage = array(testImage).reshape((len(testImage), 1))

testLabels, totalTime, times = predict(
    currentPath, testImage, bestModel, classNames)


def plotImages(figure):
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    return expand_dims(_image.decode_png(buf.getvalue(), channels=4), 0)


fileWriter = create_file_writer(logDir)
fileWriter.set_as_default()
with fileWriter.as_default():
    image('Test:\nTotal time: '+str(totalTime)+'\nTime per image: '+str(sum(times)/len(times)),
          plotImages(imageGrid(testImage,
                               testLabels,
                               classNames=classNames,
                               perRow=4,
                               imageDimensions=(32, 32),
                               nImages=16)),
          step=0)
