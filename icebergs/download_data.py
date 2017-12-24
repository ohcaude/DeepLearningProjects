import os
import subprocess

fname='test.json'
username='ocaudevi'
password='eO7MYdZ4Kni5'
competition='statoil-iceberg-classifier-challenge'
if not os.path.isfile('data/'+fname):
    print('Downloading '+fname)
    subprocess.run(['kg', 'download', '-u', username, '-p', password ,'-c', competition,'-f',fname+'.7z'])
    subprocess.run(['7z', 'e', fname+'.7z', '-o./data/'])
    subprocess.run(['rm', fname+'.7z'])
else:
    print(fname + ' was already downloaded')

fname='train.json'
if not os.path.isfile('data/'+fname):
    print('Downloading ' + fname)
    subprocess.run(['kg', 'download', '-u', username, '-p', password ,'-c', competition,'-f',fname+'.7z'])
    subprocess.run(['7z', 'e', fname+'.7z', '-o./data/'])
    subprocess.run(['rm', fname+'.7z'])
else:
    print(fname + ' was already downloaded')

