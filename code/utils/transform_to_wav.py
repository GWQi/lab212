import os

src='/home/guwenqi/Documents/audio/music/newage/'
dst='/home/guwenqi/Documents/audio_wav_channel_1/music/newage/'


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
        try:
            os.system("ffmpeg -i {} -ac 1 {}".format(path, path.replace(src, dst).replace(ext, 'wav')))
            print("transform {}'th file done!".format(i))
            i = i+1
        except:
            pass
