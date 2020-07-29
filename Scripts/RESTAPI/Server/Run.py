from os.path import join, realpath, dirname
from json import load

from uvicorn import run

from App import app


if __name__ == '__main__':
    with open(join(dirname(realpath(__file__)), 'Settings.json'), 'r') as f:
        data = load(f)
        run("App:app", host=str(data['Host']),
            port=int(data['Port']), log_level="info")
