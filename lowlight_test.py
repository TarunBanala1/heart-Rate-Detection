import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import cv2
import matplotlib.pyplot as plt
from pathlib import Path



def lowlight(frame):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    #data_lowlight = Image.open(image_path)

 

#    data_lowlight = (np.asarray(data_lowlight)/255.0)
#    #print("shape 1",data_lowlight.shape)
#
#
#    data_lowlight = torch.from_numpy(data_lowlight).float()
#    data_lowlight = data_lowlight.permute(2,0,1)
#    data_lowlight = data_lowlight.cuda().unsqueeze(0)
    
    #print("shape 2",data_lowlight.shape)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
    start = time.time()
    _,enhanced_image,_ = DCE_net(frame)
    
    return enhanced_image
#    print("shape en",enhanced_image.shape)
#
#    end_time = (time.time() - start)
#    print(end_time)
#    image_path = image_path.replace('test_data','result')
#    image_path = image_path.replace('\\' , '/')
#    result_path = image_path
#    print(image_path)
#    print(image_path.replace('/'+image_path.split('/')[-1],''))
#    if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
#    	os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
#
#    torchvision.utils.save_image(enhanced_image, image_path)
    #plt.imsave(result_path, enhanced_image[0].permute(1,2,0).cpu().numpy())
#    plt.imshow(enhanced_image[0].permute(1,2,0).cpu().numpy())
#    plt.show()
#    print('a')
    
def save_video(video_tensor, name):
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height,width]=video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter(name, -1, 30, (width, height))
    for i in range(0,video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i] * 255.0))
    writer.release()

if __name__ == '__main__':
    with torch.no_grad():
        #filePath = 'G:/ML Projects/ISI/Motion Microscopy/Baby Sleep Videos/trimmed/38.mp4'
        #filePath = 'G:/ML Projects/ISI\Motion Microscopy/stable videos/low_light/riju_4.mp4'
        filePath = 'G:/ML Projects/ISI/Motion Microscopy/stable videos/ISI/low_light'
        
        
        for lowlight_video in sorted(os.listdir(filePath)):
            video_path = os.path.join(filePath, lowlight_video)
#            video_path = 'G:/ML Projects/ISI/Motion Microscopy/Artificial Illumination/Zero-DCE-master/outputs/ISI/sub_5_en.mp4'
            video_name = lowlight_video.split('.')[0]
            output_path = 'G:/ML Projects/ISI/Motion Microscopy/Artificial Illumination/Zero-DCE-master/outputs/ISI/'+ video_name+ '_en.mp4'

            #print(video_name)
            if not os.path.exists(output_path):
                print('Processing: ', lowlight_video)
                
                cap=cv2.VideoCapture(video_path)
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # print("Video Resolution: ", frame_count, height, width)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                #video_tensor = np.zeros((frame_count, height, width,3), dtype=np.uint8)
        #        enhanced_video = np.zeros((frame_count, height, width,3))
                enhanced_video = np.zeros((frame_count, 240, 320, 3))
                
                x = 0
                
                while cap.isOpened():
                  ret,frame=cap.read()
                  if ret is True:
                      frame_lowlight = cv2.resize(frame/255.0, (320,240))
                      #print('f_l shape: ', frame_lowlight.shape)
                      
                      frame_lowlight = torch.from_numpy(frame_lowlight).float()
                      frame_lowlight = frame_lowlight.permute(2,0,1)
                      frame_lowlight = frame_lowlight.cuda().unsqueeze(0)
                      
                      frame_enhanced = lowlight(frame_lowlight)
                      #print(frame_enhanced[0].permute(1,2,0).cpu().numpy().shape)
                      enhanced_video[x] = frame_enhanced[0].permute(1,2,0).cpu().numpy()
                      #out.write(frame_enhanced[0].permute(1,2,0).cpu().numpy())
        #              plt.imshow(enhanced_video[x])
        #              plt.show()
                      x+=1
                      print("Processed frames: %d out of %d"%(x, frame_count), flush = True)
                  else:
                      break
                  
                  
                output_path = 'G:/ML Projects/ISI/Motion Microscopy/Artificial Illumination/Zero-DCE-master/outputs/ISI/'+ video_name+ '_en.mp4'
                  
                cap.release()
                save_video(enhanced_video, output_path)
          
        #print("Video Dimension: ", video_tensor.shape)
        
        
        
        
        
#        filePath = 'data/test_data/'
#        file_list = os.listdir(filePath)
#        
#        for file_name in file_list:
#            test_list = glob.glob(filePath+file_name+"/*")
#            for image in test_list:
#                print(image)
#                lowlight(image)
