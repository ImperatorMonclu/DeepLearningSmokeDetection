#!bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source $DIR/../.env/bin/activate
    $DIR/../.env/bin/python $DIR/Training/Run.py
elif [[ "$OSTYPE" == "darwin"* ]]; then
    source $DIR/../.env/bin/activate
    $DIR/../.env/bin/python $DIR/Training/Run.py
elif [[ "$OSTYPE" == "cygwin" ]]; then
    source $DIR\\..\\.env\\Scripts\\activate
    $DIR\\..\\.env\\Scripts\\python $DIR\\Training\\Run.py
elif [[ "$OSTYPE" == "msys" ]]; then
    source $DIR\\..\\.env\\Scripts\\activate
    $DIR\\..\\.env\\Scripts\\python $DIR\\Training\\Run.py
elif [[ "$OSTYPE" == "win32" ]]; then
    source $DIR\\..\\.env\\Scripts\\activate
    $DIR\\..\\.env\\Scripts\\python $DIR\\Training\\Run.py
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    source $DIR/../.env/bin/activate
    $DIR/../.env/bin/python $DIR/Training/Run.py
else
    echo "Unknown OS."
fi
