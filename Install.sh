#!bin/bash
pip install --upgrade --user pip
pip install virtualenv
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
virtualenv $DIR/.env
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source $DIR/.env/bin/activate
    $DIR/.env/bin/pip install Django==3.0.3
    $DIR/.env/bin/pip install uvicorn==0.11.3
    $DIR/.env/bin/pip install fastapi==0.46.0
    $DIR/.env/bin/pip install opencv-python==4.3.0.36
    $DIR/.env/bin/pip install imgaug==0.4.0
    $DIR/.env/bin/pip install --upgrade tensorflow-gpu==2.1.0
    $DIR/.env/bin/pip install segmentation-models==1.0.1
    $DIR/.env/bin/pip install scikit-learn==0.22.1
elif [[ "$OSTYPE" == "darwin"* ]]; then
    source $DIR/.env/bin/activate
    $DIR/.env/bin/pip install Django==3.0.3
    $DIR/.env/bin/pip install uvicorn==0.11.3
    $DIR/.env/bin/pip install fastapi==0.46.0
    $DIR/.env/bin/pip install opencv-python==4.3.0.36
    $DIR/.env/bin/pip install imgaug==0.4.0
    $DIR/.env/bin/pip install --upgrade tensorflow-gpu==2.1.0
    $DIR/.env/bin/pip install segmentation-models==1.0.1
    $DIR/.env/bin/pip install scikit-learn==0.22.1
elif [[ "$OSTYPE" == "cygwin" ]]; then
    source $DIR\\.env\\Scripts\\activate
    $DIR\\.env\\Scripts\\pip install Django==3.0.3
    $DIR\\.env\\Scripts\\pip install uvicorn==0.11.3
    $DIR\\.env\\Scripts\\pip install fastapi==0.46.0
    $DIR\\.env\\Scripts\\pip install opencv-python==4.3.0.36
    $DIR\\.env\\Scripts\\pip install imgaug==0.4.0
    $DIR\\.env\\Scripts\\pip install --upgrade tensorflow-gpu==2.1.0
    $DIR\\.env\\Scripts\\pip install segmentation-models==1.0.1
    $DIR\\.env\\Scripts\\pip install scikit-learn==0.22.1
elif [[ "$OSTYPE" == "msys" ]]; then
    source $DIR\\.env\\Scripts\\activate
    $DIR\\.env\\Scripts\\pip install Django==3.0.3
    $DIR\\.env\\Scripts\\pip install uvicorn==0.11.3
    $DIR\\.env\\Scripts\\pip install fastapi==0.46.0
    $DIR\\.env\\Scripts\\pip install opencv-python==4.3.0.36
    $DIR\\.env\\Scripts\\pip install imgaug==0.4.0
    $DIR\\.env\\Scripts\\pip install --upgrade tensorflow-gpu==2.1.0
    $DIR\\.env\\Scripts\\pip install segmentation-models==1.0.1
    $DIR\\.env\\Scripts\\pip install scikit-learn==0.22.1
elif [[ "$OSTYPE" == "win32" ]]; then
    source $DIR\\.env\\Scripts\\activate
    $DIR\\.env\\Scripts\\pip install Django==3.0.3
    $DIR\\.env\\Scripts\\pip install uvicorn==0.11.3
    $DIR\\.env\\Scripts\\pip install fastapi==0.46.0
    $DIR\\.env\\Scripts\\pip install opencv-python==4.3.0.36
    $DIR\\.env\\Scripts\\pip install imgaug==0.4.0
    $DIR\\.env\\Scripts\\pip install --upgrade tensorflow-gpu==2.1.0
    $DIR\\.env\\Scripts\\pip install segmentation-models==1.0.1
    $DIR\\.env\\Scripts\\pip install scikit-learn==0.22.1
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    source $DIR/.env/bin/activate
    $DIR/.env/bin/pip install Django==3.0.3
    $DIR/.env/bin/pip install uvicorn==0.11.3
    $DIR/.env/bin/pip install fastapi==0.46.0
    $DIR/.env/bin/pip install opencv-python==4.3.0.36
    $DIR/.env/bin/pip install imgaug==0.4.0
    $DIR/.env/bin/pip install --upgrade tensorflow-gpu==2.1.0
    $DIR/.env/bin/pip install segmentation-models==1.0.1
    $DIR/.env/bin/pip install scikit-learn==0.22.1
else
    echo "Unknown OS."
fi
