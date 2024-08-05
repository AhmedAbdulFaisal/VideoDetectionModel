from pytubefix import YouTube
import shutil


#uses pytubefix, which is the better version of pytube
# Doccumentation: 
#
# download_video(url): downloads video and places it in video-assignment
#   *arguments: string - url: video url for downloading
# set_captions(url): downloads captions and places it in video-assignmemt
#   *arguments: string - url: video url for downloading
#
# uses pre-set path as shown below. 
#
#




#we have a non-relative file path here
path = '/home/ahmed/projects/cs370hw/cs370-assignments/video-assignment'

def download_video(url,name):
    vid=YouTube(url)
    print(vid.title)
    vid.bypass_age_gate()
    vid.streams.first().download(path, name)

def set_captions(url, name):
    vid=YouTube(url)
    caption = vid.captions.get_by_language_code('a.en')
    caption.save_captions(name + ".txt")
    shutil.move(name + ".txt", path + "/" + name + ".txt")

download_video('https://www.youtube.com/watch?v=wbWRWeVe1XE', 'video1.mp4')
set_captions('https://www.youtube.com/watch?v=wbWRWeVe1XE', 'video1')

download_video('https://www.youtube.com/watch?v=FlJoBhLnqko', 'video2.mp4')
set_captions('https://www.youtube.com/watch?v=FlJoBhLnqko', 'video2')

download_video('https://www.youtube.com/watch?v=Y-bVwPRy_no', 'video3.mp4')
set_captions('https://www.youtube.com/watch?v=Y-bVwPRy_no', 'video3')

#video 1

# vid=YouTube('https://www.youtube.com/watch?v=wbWRWeVe1XE')
# print(vid.title)
# vid.bypass_age_gate()
# vid.streams.first().download(path)

# caption = vid.captions.get_by_language_code('a.en')
# caption.save_captions(vid.title + ".txt")
# shutil.move(vid.title + ".txt", path + "/" + vid.title + ".txt")

# #video 2

# vid=YouTube('https://www.youtube.com/watch?v=FlJoBhLnqko')
# print(vid.title)
# vid.bypass_age_gate()
# vid.streams.first().download(path)

# caption = vid.captions.get_by_language_code('a.en')
# caption.save_captions(vid.title + ".txt")
# shutil.move(vid.title + ".txt", path + "/" + vid.title + ".txt")

# #video 3

# vid=YouTube('https://www.youtube.com/watch?v=Y-bVwPRy_no')
# print(vid.title)
# vid.bypass_age_gate()
# vid.streams.first().download(path)

# caption = vid.captions.get_by_language_code('a.en')
# caption.save_captions(vid.title + ".txt")
# shutil.move(vid.title + ".txt", path + "/" + vid.title + ".txt")
