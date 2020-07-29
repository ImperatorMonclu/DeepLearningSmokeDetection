from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow import expand_dims, image

from os.path import join
from os import sep
from glob import glob
from tensorflow.keras.models import load_model


# Guarda la imagen en matplotlib
def plotImages(figure):
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    return expand_dims(image.decode_png(buf.getvalue(), channels=4), 0)


# Devuelve el modelo cargado con los mejores pesos
def bestCheckpointLoad(relativePath, checkpointsPath):
    checkpointFilenames = list(glob(join(checkpointsPath, '*')))
    xMax = 0.0
    for f in checkpointFilenames:
        x = float(f.split(sep)[-1][-9:-3])
        if x > xMax:
            xMax = x
            checkpoint = f
    model = load_model(join(relativePath, 'Model.h5'))
    model.load_weights(checkpoint)
    return model
