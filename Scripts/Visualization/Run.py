from sys import stdout
from json import load
from os.path import join, realpath, dirname

with open(join(dirname(realpath(__file__)), 'Settings.json'), 'r') as f:
    data = load(f)
    if bool(data['Debug']):
        stdout.write(str(data['Relative'])+' --host '+str(data['Host'])+' --port '+str(
            int(data['Port']))+' --debugger_port '+str(int(data['DebugPort'])))
    else:
        stdout.write(str(data['Relative'])+' --host ' +
                     str(data['Host']) + ' --port '+str(int(data['Port'])))
