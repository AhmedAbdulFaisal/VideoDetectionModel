# VideoDetectionModel
Fully working VideoDetection using a pre-trained MS-COCO model I made for an assignment during my 'Intro to AI' class. The videos referenced within the code must be stored within the root directory, as they are not supplied in the repository.

The model utilized is fasterrcnn_resnet50 from Pytorch torchvision. Detection was done on a sample of three videos featuring static objects like furniture and moving subjects like people and vehicles. Despite being somewhat slow on my machine. This version only spits out a given list of objects, later versions will be able to show a visual image (in matplotlib) with the detected objects. Unfortunately I was never able to get opencv working with this. 


