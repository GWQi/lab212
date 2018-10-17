#! /usr/bin/python
# -*- coding:utf-8 -*-

# 使用youtube-dl爬取youtube视频

import os
import sys
import time
import subprocess


def check_format_index(url):
	"""根据视频地址检查可用的音视频格式"""
	
    cmd_check_audio = "youtube-dl --proxy socks5://127.0.0.1:1080 -F " + url + " 2>&1|grep '^139' -m 1 | awk '{print $1}'"  # 检测音频格式索引
    cmd_check_video = "youtube-dl --proxy socks5://127.0.0.1:1080 -F " + url + " 2>&1|grep '^160' -m 1 | awk '{print $1}'"  # 检测视频格式索引
    audio_index = subprocess.check_output(cmd_check_audio, shell = True).strip()
    video_index = subprocess.check_output(cmd_check_video, shell = True).strip()
    if audio_index and video_index:
        return '160+139'  # 默认抓取的视频格式索引为160，抓取的音频格式索引为139
    else:
        cmd_check_3gp = "youtube-dl --proxy socks5://127.0.0.1:1080 -F " + url + " |grep 3gp -m 1 | awk '{print $1}'"
        index_3gp = subprocess.check_output(cmd_check_3gp, shell = True).strip()  # 若存在3gp格式的视频格式则使用3gp格式
        return index_3gp if index_3gp else False


def youtube_dl(index_file):
    with open(index_file, 'r+') as f:
        succeed_counts = 0
        curr_folder = 10
        for line in f:
            video_id = line.split(', ')[0]
            failed_times = 0
            if not subprocess.call("grep " + video_id + " download_completed", shell=True):
                continue  # 若当前抓取的任务在已完成的列表中则跳过
            else:                
                url = 'http://www.youtube.com/watch?v=' + video_id
                format_index = check_format_index(url)  # 获取可用的音视频格式索引
                if format_index:
                    ext = '.mp4' if format_index == '160+139' else '.3gp'
                    save_folder = './mp4/' + str(curr_folder)
                    if succeed_counts == 1000:
                        sys.exit(1)  # 若完成任务数达到1000则退出程序
                    save_name = save_folder + '/' + video_id + ext
                    cmd_download = 'youtube-dl --proxy socks5://127.0.0.1:1080 -f ' + format_index + ' ' + url + ' -i -o ' + save_name
                    return_code = subprocess.call(cmd_download, shell = True)
                    if not return_code:
                        with open('download_completed', 'a') as fp:
                            fp.write(video_id + '\n')  # 若当前任务抓取完成则成功数加一
                            succeed_counts += 1
                    while return_code:
                        failed_times += 1
                        return_code = subprocess.call(cmd_download, shell = True)
                        if not return_code:
                            with open('download_completed', 'a') as fp:
                                fp.write(video_id + '\n')
                            succeed_counts += 1
                            break
                        if failed_times == 4:  # 若当前任务连续四次抓取失败则跳过此任务
                            subprocess.call('rm ./mp4/' + str(curr_folder) + '/'+ video_id + '*', shell = True)
                            break
                    time.sleep(0.5)
                else:
                    continue


if __name__ == '__main__':
    index_file = 'balanced_train_segments.csv'
    youtube_dl(index_file)