#!bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source $DIR/../.env/bin/activate
    configTensorBoard=$($DIR/../.env/bin/python $DIR/Visualization/Run.py)
    $DIR/../.env/bin/tensorboard --logdir $configTensorBoard
elif [[ "$OSTYPE" == "darwin"* ]]; then
    source $DIR/../.env/bin/activate
    configTensorBoard=$($DIR/../.env/bin/python $DIR/Visualization/Run.py)
    $DIR/../.env/bin/tensorboard --logdir $configTensorBoard
elif [[ "$OSTYPE" == "cygwin" ]]; then
    source $DIR\\..\\.env\\Scripts\\activate
    configTensorBoard=$($DIR\\..\\.env\\Scripts\\python $DIR\\Visualization\\Run.py)
    $DIR\\..\\.env\\Scripts\\tensorboard --logdir $configTensorBoard
elif [[ "$OSTYPE" == "msys" ]]; then
    source $DIR\\..\\.env\\Scripts\\activate
    configTensorBoard=$($DIR\\..\\.env\\Scripts\\python $DIR\\Visualization\\Run.py)
    $DIR\\..\\.env\\Scripts\\tensorboard --logdir $configTensorBoard
elif [[ "$OSTYPE" == "win32" ]]; then
    source $DIR\\..\\.env\\Scripts\\activate
    configTensorBoard=$($DIR\\..\\.env\\Scripts\\python $DIR\\Visualization\\Run.py)
    $DIR\\..\\.env\\Scripts\\tensorboard --logdir $configTensorBoard
elif [[ "$OSTYPE" == "freebsd"* ]]; then
    source $DIR/../.env/bin/activate
    configTensorBoard=$($DIR/../.env/bin/python $DIR/Visualization/Run.py)
    $DIR/../.env/bin/tensorboard --logdir $configTensorBoard
else
    echo "Unknown OS."
fi
