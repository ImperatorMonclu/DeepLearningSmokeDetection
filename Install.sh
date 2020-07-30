#!bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHONDIR="$(which python)"
pip install --upgrade --user pip
pip install virtualenv
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
    sudo cp -f $DIR/Scripts/Lib/mimetypes.py "$(dirname "${PYTHONDIR}")"/Lib
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
    sudo cp -f $DIR/Scripts/Lib/mimetypes.py "$(dirname "${PYTHONDIR}")"/Lib
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
    WINPYTHONC=$(dirname "${PYTHONDIR}")
    WINPYTHON=${WINPYTHONC:2}
    WINDIR=${DIR:2}
    echo "copy /Y "${WINDIR//\//\\}"\\Scripts\\Lib\\mimetypes.py "${WINPYTHON//\//\\}"\\Lib""" >WindowsFix.bat
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
    WINPYTHONC=$(dirname "${PYTHONDIR}")
    WINPYTHON=${WINPYTHONC:2}
    WINDIR=${DIR:2}
    echo "copy /Y "${WINDIR//\//\\}"\\Scripts\\Lib\\mimetypes.py "${WINPYTHON//\//\\}"\\Lib""" >WindowsFix.bat
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
    WINPYTHONC=$(dirname "${PYTHONDIR}")
    WINPYTHON=${WINPYTHONC:2}
    WINDIR=${DIR:2}
    echo "copy /Y "${WINDIR//\//\\}"\\Scripts\\Lib\\mimetypes.py "${WINPYTHON//\//\\}"\\Lib""" >WindowsFix.bat
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
    sudo cp -f $DIR/Scripts/Lib/mimetypes.py "$(dirname "${PYTHONDIR}")"/Lib
else
    echo "Unknown OS."
fi
