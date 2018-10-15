import os
import shutil
import librosa as lrs
from scipy.io import wavfile

src='/home/guwenqi/Documents/ASC-test/speech'
dst='/home/guwenqi/Documents/ASC-test/speech_ds'
torate=16000

for root, dirlist, filelist in os.walk(src):
    try:
        os.makedirs(root.replace(src, dst))
    except:
        pass


i = 1
for root, dirlist, filelist in os.walk(src):
    for file in filelist:
        ext = file.split('.')[-1]
        path = os.path.join(root, file)
        if ext == 'wav':
            try:
                data, _ = lrs.load(path, sr=torate, mono=True)
            except:
                print('Please check if {} is a readable wav file'.format(path))
            wavfile.write(path.replace(src, dst).replace(ext, 'wav'), torate, data)
            print("resample {}'th file done!".format(i))
            i = i+1
        elif ext == 'lab':
            shutil.copy(path, path.replace(src, dst))

