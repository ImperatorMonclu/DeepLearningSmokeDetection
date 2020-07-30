# Detección de humo a partir de técnicas de Deep Learning

## Características

- Entrenamiento de redes neuronales.
- Visualización a partir de TensorBoard.
- API REST funcional dedicada a predicciones masivas tanto de clasificación como de segmentación de imágenes.
- Despliegue de aplicación web.
- Despliegue de detector de humo en una Raspberry Pi.
- Multiplataforma.

## Requisitos

- [Python 3.7.*](https://www.python.org/downloads/)
- Terminal Bash (Para Windows, [Git](https://git-scm.com/downloads) instala opcionalmente una terminal Bash)

Para utilizar aceleración por GPU:
- Tarjeta gráfica de NVIDIA con controlador 418.* o superior
- CUDA 10.1
- cuDNN 7.6
- Mas información [aquí](https://www.tensorflow.org/install/gpu)

## Instalación

  - Ejecutar el archivo Install.sh del directorio principal
```sh
$ bash Install.sh
```
        
### Windows

Después de la instalación se generará un archivo WindowsFix.bat que se deberá "Ejecutar como administrador" para poder ejecutar la aplicación web y la visualización de resultados correctamente.

## Datos opcionales

Se tiene a disposición modelos ya cargados, imágenes y estructura de archivos en [este enlace](https://drive.google.com/file/d/1EBkV-XS7eZRriETU4EEcbr6wgAo7N6hH/view?usp=sharing).

## Ejecución

Excepto para Raspberry Pi, la ejecución de las aplicaciones son en la carpeta Scripts.

### Entrenamiento

Scripts/Training

Para su configuración se puede variar en Settings.json y Settings.py. En Settings.json la variable RelativeData será una dirección que deberá de seguir un tipo de estructura de carpetas variando según si es clasificación o segmentación.

#### Clasificación
Datos de entrenamiento:
- RelativeData/Train/clase1
- RelativeData/Train/clase2
- ...

Datos de validación:
- RelativeData/Validation/clase1
- RelativeData/Validation/clase2
- ...

Datos de prueba: RelativeData/Test

#### Segmentación
Datos de entrenamiento:
- RelativeData/Train/Images
- RelativeData/Train/Labels/clase1
- RelativeData/Train/Labels/clase2
- ...

Datos de validación:
- RelativeData/Validation/Images
- RelativeData/Validation/Labels/clase1
- RelativeData/Validation/Labels/clase2
- ...

Datos de prueba: RelativeData/Test

La ejecución se realizará con la ejecución de Scripts/Training.sh
```sh
$ bash Scripts/Training.sh
```

### Visualización

Scripts/Visualization

Para su configuración se puede variar en Settings.json. La variable Relative deberá ser la dirección de un modelo entrenado o en entrenamiento.

La ejecución se realizará con la ejecución de Scripts/Visualization.sh
```sh
$ bash Scripts/Visualization.sh
```

### Website

Scripts/Website

Para su configuración se puede variar en Settings.json y Settings.sh. En Settings.json la variable Relative será una dirección que deberá de seguir un tipo de estructura de carpetas.

Modelos de clasificación: Relative/Models/Classification

Modelos de segmentación: Relative/Models/Segmentation

La ejecución se realizará con la ejecución de Scripts/Website.sh
```sh
$ bash Scripts/Website.sh
```

### API REST Servidor

Scripts/RESTAPI/Server

Para su configuración se puede variar en Settings.json. La variable Relative será una dirección que deberá de seguir un tipo de estructura de carpetas.

Modelos: Relative/Models

La ejecución se realizará con la ejecución de Scripts/RESTAPIServer.sh
```sh
$ bash Scripts/RESTAPIServer.sh
```

### API REST Cliente

Scripts/RESTAPI/Client

Para su configuración se puede variar en Settings.json. La variable Relative será una dirección que deberá de seguir un tipo de estructura de carpetas.

Imágenes: Relative/Images

La ejecución se realizará con la ejecución de Scripts/RESTAPIClient.sh
```sh
$ bash Scripts/RESTAPIClient.sh
```

### Raspberry Pi

#### Instalación

Instalar el sistema operativo [dietPi](https://dietpi.com/phpbb/viewtopic.php?p=9#p9).

A continuación, con un ordenador a parte se deberá copiar la carpeta RaspberryPi/Smoke en un pen drive USB directamente en el directorio raíz (Ejemplo F:/Smoke) y conectarlo a la Raspberry Pi.

Ejecutar en la shell de la Raspberry Pi los siguientes comandos en el orden que corresponde:
```sh
$ cd /root
$ sudo wget https://gist.githubusercontent.com/MauroGarciaMonclu/731e887fe7a7f2c71c345f34e66d16ac/raw/15b139f9a93a67250a47ebd50a8e7250acc501ca/Install.sh
$ sudo bash /root/Install.sh
```

Conectar cámaras USB.

Para realizar una predicción (Y apagado si está configurado) ejecutar:
```sh
$ sudo bash /root/Smoke/Run.sh
```

Para realizar una prueba de mantenimiento, primero conectar un pen drive USB y a continuación ejecutar:
```sh
$ sudo bash /root/Smoke/Test.sh
```