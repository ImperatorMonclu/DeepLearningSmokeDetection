import imgaug.augmenters as iaa

import tensorflow.keras.optimizers as tfko

import tensorflow.keras.callbacks as tfkc

from datetime import datetime
from os.path import join, realpath, dirname
from json import load

# Lee el archivo de configuración
with open(join(dirname(realpath(__file__)), 'Settings.json'), 'r') as f:
    data = load(f)
    relative = join(str(data['Relative']))
    namePrefix = str(data['NamePrefix'])
    classNames = data['ClassNames']
    learningRate = float(data['LearningRate'])
    mode = str(data['Mode'])
    saveBest = bool(data['SaveBest'])


# Optimizador 
optimizer = tfko.Adam(learning_rate=learningRate)

# Data augmentation
augmenters = iaa.Sequential(  # iaa.Sequential() or None
    [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.125),
        iaa.Multiply((0.625, 1.375),
                     per_channel=False),
        iaa.Multiply((0.875, 1.125),
                     per_channel=0.25),
        iaa.Sometimes(0.875,
                      [
                          iaa.Affine(scale={'x': (0.75, 1.25),
                                            'y': (0.75, 1.25)},
                                     cval=0,
                                     mode='constant')
                      ]
                      ),
        iaa.Sometimes(0.875,
                      [
                          iaa.Affine(translate_percent={
                              'x': (-0.25, 0.25),
                              'y': (-0.25, 0.25)},
                              cval=0,
                              mode='constant')
                      ]
                      )
    ],
    random_order=False)

# Horario learning rate
'''
def LRSchedule(epoch):
    LearningRate = 0.001
    if epoch > 180:
        LearningRate *= 0.0005
    elif epoch > 160:
        LearningRate *= 0.001
    elif epoch > 120:
        LearningRate *= 0.01
    elif epoch > 80:
        LearningRate *= 0.1
    return LearningRate
'''

# Nombres de las carpetas
currentPath = join(relative, namePrefix +
                   datetime.now().strftime('%Y%m%d%H%M%S'))
logDir = join(currentPath, 'Log')
monitor = ('val_binary_accuracy' if len(classNames) ==
           2 else 'val_categorical_accuracy') if mode == 'classification' else 'val_iou_score'
checkpointsDir = join(currentPath, 'Checkpoints')

# Callbacks
callbacks = [
    tfkc.ModelCheckpoint(filepath=join(checkpointsDir,
                                       'Epoch{epoch:04d}Accuracy{'+monitor+':.4f}.h5'),
                         monitor=monitor,
                         mode='max',
                         save_weights_only=True,
                         save_best_only=saveBest,
                         verbose=1),
    tfkc.ReduceLROnPlateau(monitor=monitor,
                           mode='max',
                           factor=0.5,
                           patience=4,
                           cooldown=4,
                           min_lr=0.0,
                           min_delta=0.0,
                           verbose=1),
    tfkc.ReduceLROnPlateau(monitor=monitor,
                           mode='max',
                           factor=0.5,
                           patience=16,
                           cooldown=0,
                           min_lr=0.0,
                           min_delta=0.0,
                           verbose=1),
    # tfkc.LearningRateScheduler(LRSchedule,
    #                           verbose=1),
    tfkc.TensorBoard(log_dir=logDir,
                     histogram_freq=1,
                     profile_batch=0,
                     write_graph=True,
                     write_images=True,
                     update_freq='epoch',
                     embeddings_freq=0,
                     embeddings_metadata=None)
]
