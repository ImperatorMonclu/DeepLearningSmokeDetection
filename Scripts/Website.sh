#!bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source $DIR/../.env/bin/activate
    source $DIR/Website/Settings.sh
    $DIR/../.env/bin/python $DIR/Website/manage.py runserver $Host:$Port
elif [[ "$OSTYPE" == "darwin"* ]]; then
    source $DIR/../.env/bin/activate
    source $DIR/Website/Settings.sh
    $DIR/../.env/bin/python $DIR/Website/manage.py runserver $Host:$Port
elif [[ "$OSTYPE" == "cygwin" ]]; then
    source $DIR\\..\\.env\\Scripts\\activate
    source $DIR\\Website\\Settings.sh
    $DIR\\..\\.env\\Scripts\\python $DIR\\Website\\manage.py runserver $Host:$Port
elif [[ "$OSTYPE" == "msys" ]]; then
    source $DIR\\..\\.env\\Scripts\\activate
    source $DIR\\Website\\Settings.sh
    $DIR\\..\\.env\\Scripts\\python $DIR\\Website\\manage.py runserver $Host:$Port
elif [[ "$OSTYPE" == "win32" ]]; then
    source $DIR\\..\\.env\\Scripts\\activate
    source $DIR\\Website\\Settings.sh
    $DIR\\..\\.env\\Scripts\\python $DIR\\Website\\manage.py runserver $Host:$Port
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    source $DIR/../.env/bin/activate
    source $DIR/Website/Settings.sh
    $DIR/../.env/bin/python $DIR/Website/manage.py runserver $Host:$Port
else
    echo "Unknown OS."
fi
