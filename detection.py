import torchvision.transforms as transforms
import torchvision
import torch
import argparse 
import cv2
import numpy as np
from PIL import Image
import preprocess as prep
import math

# DETECTION.PY
# BY AHMED FAISAL
# CLASS IS DESIGNED TO TAKE DATA FROM PREPROCESS.PY (.npy binary), AND THEN RUN IT THROUGH THE AUTODETECTOR
# THE AUTODETECTOR THEN RUNS IT THROUGH AND DETECTS ALL THE POSSIBLE OBJECTS IN THE SCENE
# THE OBJECTS ARE THEN TABULATED AS PRINTED STATEMENTS 
# ALL GPU-SPECIFIC FUNCTIONS ARE HANDLED BY THE CPU DUE TO THE WORKSTATION NOT HAVING SUITABLE HARDWARE FOR GPU ACCELERATION
# (I'm pretty sure it works, idk it seems to detect the school just fine ig)

#instance names for COCO recognition
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COLORS = np.random.uniform(0,256,size=(len(COCO_INSTANCE_CATEGORY_NAMES),3))

transform = transforms.Compose([
    transforms.ToTensor(),
])

#path = '/content/drive/MyDrive/Data-npy/frames.npy'






def model_initialize(path):
  #parser = argparse.ArgumentParser()
  #parser.add_argument('-i', '--input', help='path to input image/video')
  #parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
  #                  help='minimum input size for the FasterRCNN network')
  #args = vars(parser.parse_args())

  print("initializing model")
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,min_size=800)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.eval().to(device)


  print("the path is" + ' ' + path)
  frames = prep.process_frames(path)
 


  frame_counter=0
  for frame_sample in frames:
    #print("processing frame... please wait...")
    boxes,classes,labels = predict(frame_sample,model,device,0.8, frame_counter, '01')
    resulting_image=draw_boxes(boxes,classes,labels,frame_sample)
    #print(frame_counter)
    frame_counter+=1
    


  #print(path + '' + "/video1.mp4")
  #images = prep.process_frames(path+'/video1.mp4', 100)
  #print("done")
  #model.eval().to(device)

  #boxes,classes,labels = predict(images[50],model,device,0.8)
  #image=draw_boxes(boxes,classes,labels,images[50])
  #cv2.imshow('Image',image)
  #cv2.waitKey(0)

  #cv2.imshow('Image',sample_frame)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()


def predict(image, model, device, threshold, number,vidID):
  image = transform(image).to(device)
  image = image.unsqueeze(0)
  outputs=model(image)

  pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in outputs[0]['labels'].cpu().numpy()]

  pred_scores = outputs[0]['scores'].detach().cpu().numpy()

  pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

  boxes=pred_bboxes[pred_scores >= threshold].astype(np.int32)


  #print data here, as:
  #[index,exactFrame,detectedObjects]
  print("%s | %s | %s | %s" % (vidID,number,number*30,pred_classes))



  return boxes,pred_classes,outputs[0]['labels']

def draw_boxes(boxes,classes,labels,img):
  img=cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
  
  for i,box in enumerate(boxes):
    color = COLORS[labels[i]]
    cv2.rectangle(
        img,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        color, 2
    )
    cv2.putText(img,classes[i], (int(box[0]), int(box[1]-5)),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2,
                lineType=cv2.LINE_AA)
                
    return img
  


model_initialize('/home/ahmed/projects/cs370hw/cs370-assignments/video-assignment/video1.mp4')
