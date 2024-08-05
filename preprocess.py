import cv2
import numpy as np
import math

# PREPROCESS.PY
# BY AHMED FAISAL
# THIS CLASS TAKES THE VIDEO, SAMPLES IT BY 30 SECONDS, AND THEN PROCESSES IT AS A .NPY FILE
# VARIOUS HELPER AND DEBUG CLASSES ARE SHOWN TO TEST THE FEATURE
# Decode Frames
# Sample Frames
# Resize images
# Scaling, Normalizing, and Correcting Color Mode
# Save to .npy

path = '/home/ahmed/projects/cs370hw/cs370-assignments/video-assignment'
pathvideo = path + "/video1/"

video1 = '/video1.mp4'
video2 = '/video2.mp4'
video3 = '/video3.mp4'

def main():
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()


#Completes the entire pipeline of preprocessing:
# First, we decode and sample (30fps per sampled frame)
# Then, we resize
# Then, we
#
#p - path of video
#view_frame - view a specific frame in the array (unused)
def process_frames(p):
    print("processing frames")
    cap = cv2.VideoCapture(p)

    #600x600, working with prehistoric svga-resolutions here
    down_width = 256
    down_height = 256
    down_points=(down_width,down_height)
    normalized_width = 256
    normalized_height = 256
    normalized_points=np.zeros((normalized_width,normalized_height))

    #initialize constants for video
    frameCount=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #we add 20 for some extra buffer space, this 20 is unused but is present to prevent the array from going out of bounds
    #buf=np.empty((math.ceil(frameCount/30)+20,frameHeight,frameWidth,3),np.dtype('uint8'))
    buf=np.empty((math.ceil(frameCount/30)+20,normalized_width,normalized_height,3),np.dtype('uint8'))

    index=0
    frames=0
    if (cap.isOpened==False):
        print("ERR: NO VIDEO")

    while cap.isOpened():
        ret,frame=cap.read()
        
        if ret:
            
            resized_image=cv2.resize(frame,down_points,interpolation=cv2.INTER_LINEAR)
            normalized_image=cv2.normalize(resized_image,normalized_points,0,255,cv2.NORM_MINMAX)
            lab=cv2.cvtColor(normalized_image,cv2.COLOR_BGR2LAB)
            
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
            l=clahe.apply(l)
            lab=cv2.merge((l,a,b))
            output_l=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

            buf[index]=output_l

            #for some reason it really does not like running this snippet of code outside preprocess.py, so we will just keep it commented out

            #cv2.namedWindow('Frame Normalized',cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('Frame Normalized', normalized_image)
            #cv2.namedWindow('Frame',cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('Frame', buf[index])
            #if cv2.waitKey(25) & 0xFF == ord('q'):
            #   break
            index+=1
            frames+=30
            cap.set(cv2.CAP_PROP_POS_FRAMES,frames)
            print(index)


        else:
            cap.release()
            break
    
    #while (frame < frameCount and ret):
    #    if (frame % 30 == 0):
    #        count+=1
    #        ret,buf[count]=cap.read()
    #        print(buf[count])
    #        print(count)
    #    frame+=1
    #cap.release()
    print(buf.size)
    
    #cv2.namedWindow('frame {:d}'.format(view_frame))
    #cv2.imshow('frame 10',buf[view_frame])
    cv2.destroyAllWindows()
    return buf

#saves video as an array of images
#p - path to video
#p_s - path to save image
def writeVideo(p, p_s):
    cap = cv2.VideoCapture(p)
    if (cap.isOpened==False):
        print("ERR: NO VIDEO")
    count=0
    while (cap.isOpened()):
        ret,frame=cap.read()
        if ret == True:
            cv2.imwrite(p_s + '/data/video1/' + 'frame{:d}.jpg'.format(count),frame)
            
            print('writing frame...')
            count+=30
            cap.set(cv2.CAP_PROP_POS_FRAMES,count)
        else:
            cap.release()
            break



def playVideo(p):
    cap = cv2.VideoCapture(p)
    if (cap.isOpened==False):
        print("ERR: NO VIDEO")

    print("PRESS q TO EXIT VIDEO")

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            cv2.namedWindow('Frame',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Frame', frame)


            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    #free resources from memory
    cap.release()
    cv2.destroyAllWindows()



    
#playVideo(path + video1)
#buffer = process_frames(path+video1, 100)
#np.save('/home/ahmed/projects/cs370hw/cs370-assignments/video-assignment/frames.npy',buffer)
#writeVideo(path + video1,path)
