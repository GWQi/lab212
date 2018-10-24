# this script is used to download audioset
import re
import os
import sys
import csv
import time
import argparse
import subprocess
import multiprocessing as mp

lock = mp.Lock()

def download_audio(yid, start, end, args):
  cmd = "youtube-dl --socket-timeout 10 -f m4a/bestaudio/aac/ogg/mp3/wav/mp4/3gp -x " +\
        "--proxy socks5://127.0.0.1:1080 " +\
        "https://www.youtube.com/watch?v=" + yid +\
        " -o '{}'".format(os.path.join(args.tmp, "%(id)s.%(ext)s"))
  # cmd = "youtube-dl --proxy socks5://127.0.0.1:1080 -f m4a/bestaudio/aac/ogg/mp3/wav " +\
  #       "https://www.youtube.com/watch?v=" + yid +\
  #       " -o '{}'".format(os.path.join(args.tmp, "%(id)s.%(ext)s"))+\
  #       " --exec 'ffmpeg -i {} -ss {} -to {} {} && rm {}'".format(
  #         os.path.join(args.tmp, "%(id)s.%(ext)s"),
  #         start,
  #         end,
  #         os.path.join(current_dir, "%(id)s.%(ext)s"),
  #         os.path.join(args.tmp, "%(id)s.%(ext)s"))
  # cmd = "youtube-dl -f m4a/bestaudio/aac/ogg/mp3/wav " +\
  #       "https://www.youtube.com/watch?v=" + yid +\
  #       " -o '{}'".format(os.path.join(args.tmp, "%(id)s.%(ext)s"))+\
  #       " --exec 'ffmpeg -i {} -y -ss %s -to %s {}'" % (start, end)
  print(yid)
  download_code = subprocess.call(cmd, shell=True)

  global lock
  if not download_code:
    time.sleep(0.5)
    # make current directory according current time under wich to save the cutted 10-seconds segments
    current_dir = os.path.join(args.root, time.strftime("%Y-%m-%d", time.localtime()))
    try:
      os.makedirs(current_dir)
    except OSError:
      pass

    for filename in os.listdir(args.tmp):
      if filename.split('.')[0] == yid:
        filepath = os.path.join(args.tmp, filename)
        cut_path = filepath.replace(args.tmp, current_dir)
        # cut the file get 10-second segments
        cut_cmd = "ffmpeg -i {} -ss {} -to {} {} && del {}".format(filepath, start, end, cut_path, filepath)
        cut_code = subprocess.call(cut_cmd, shell=True)
        # if the cut command executed by ffmpeg and rm is succefful, acquire the lock and write the yid, else remove all tmp files
        if not cut_code:
          lock.acquire()
          with open(os.path.join(args.log, "downloaded_audio_ids"), 'a') as f:
            f.write(yid+'\n')
          lock.release()
        else:
          subprocess.call("rm {}.*".format(os.path.join(args.tmp, yid)), shell=True)
  else:
    lock.acquire()
    with open(os.path.join(args.log, "attempt_but_failed_ids"), 'a') as f:
      f.write(yid+'\n')
    lock.release()

def download_audioset(args):

  downloaded_audio_ids = ""
  if os.path.exists(os.path.join(args.log, 'downloaded_audio_ids')):
    with open(os.path.join(args.log, 'downloaded_audio_ids'), 'r') as f:
      downloaded_audio_ids = "".join(f.readlines())
  else:
    with open(os.path.join(args.log, 'downloaded_audio_ids'), 'w') as f:
      pass

  attempt_but_failed_ids = ""
  if os.path.exists(os.path.join(args.log, 'attempt_but_failed_ids')):
    with open(os.path.join(args.log, 'attempt_but_failed_ids'), 'r') as f:
      attempt_but_failed_ids = "".join(f.readlines())
  else:
    with open(os.path.join(args.log, 'attempt_but_failed_ids'), 'w') as f:
      pass

  with open(args.csv, 'r') as f:
    dataset = csv.reader(f)

    pool = mp.Pool(args.workers)

    try:
      for aline in dataset:
        if aline[0][0] == '#':
          continue
        yid, start, end = aline[0], aline[1], aline[2]
        if re.search(yid, downloaded_audio_ids):
          # print(yid + " is attempted!!")
          continue
        pool.apply_async(download_audio, args=(yid.strip(), start, end, args))
    except KeyboardInterrupt:
      sys.exit(1)

    try:
      pool.close()
      pool.join()
    except KeyboardInterrupt:
      sys.exit(1)




def main():
  usage =\
  """
usage: audioset.py [-h] [-c CSV] [-t TMP] [-r ROOT] [-l LOG] [-n WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  -c CSV, --csv CSV     csv file path
  -t TMP, --tmp TMP     temperal directory path, must be absolute path
  -r ROOT, --root ROOT  root directory where to save audio files, must be
                        absolute path
  -l LOG, --log LOG     directory where to save log files, including
                        downloaded files id and so on, must be absolute path
  -n WORKERS, --workers WORKERS
                        number of threads, default is 1
  """

  script_basedir = os.path.dirname(os.path.abspath(__file__))
  parser = argparse.ArgumentParser()

  parser.add_argument("-c", "--csv", type=str, default="", help="csv file path")
  parser.add_argument("-t", "--tmp", type=str, default=os.path.join(script_basedir, "tmp"), help="temperal directory path, must be absolute path")
  parser.add_argument("-r", "--root", type=str, default=os.path.join(script_basedir, "root"), help="root directory where to save audio files, must be absolute path")
  parser.add_argument("-l", "--log", type=str, default=os.path.join(script_basedir, "log"), help="directory where to save log files, including downloaded files id and so on, must be absolute path")
  parser.add_argument("-n", "--workers", type=int, default=1, help="number of threads, default is 1")

  args = parser.parse_args()
  
  if args.csv == "":
    print("Error: You must specify a csv file")
    print(usage)
    sys.exit(1)
  if not os.path.isabs(args.tmp):
    print("Error: The tmp directory must be absolute path")
    print(usage)
    sys.exit(1)
  if not os.path.isabs(args.root):
    print("Error: The root directory must be absolute path")
    print(usage)
    sys.exit(1)
  if not os.path.isabs(args.log):
    print("Error: The log directory must be absolute path")
    print(usage)
    sys.exit(1)

  # create rerelevant directories
  try:
    os.makedirs(args.tmp)
  except:
    pass
  try:
    os.makedirs(args.root)
  except:
    pass
  try:
    os.makedirs(args.log)
  except:
    pass

  download_audioset(args)
  


if __name__ == '__main__':
  main()
