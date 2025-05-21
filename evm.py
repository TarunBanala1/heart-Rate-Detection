
import cv2
import os
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from pathlib import Path

# EVM Class
class EVM():

  def __init__(self, heart_rate = False):
    # self.video_path = video_path
    # self.mode = mode
    self.heart_rate = heart_rate
    self.mag_video = None
  
  #convert RBG to YIQ
  def rgb2ntsc(self, src):
      [rows,cols]=src.shape[:2]
      dst=np.zeros((rows,cols,3),dtype=np.float64)
      T = np.array([[0.114, 0.587, 0.298], [-0.321, -0.275, 0.596], [0.311, -0.528, 0.212]])
      for i in range(rows):
          for j in range(cols):
              dst[i, j]=np.dot(T,src[i,j])
      return dst

  #convert YIQ to RBG
  def ntsc2rbg(self, src):
      [rows, cols] = src.shape[:2]
      dst=np.zeros((rows,cols,3),dtype=np.float64)
      T = np.array([[1, -1.108, 1.705], [1, -0.272, -0.647], [1, 0.956, 0.620]])
      for i in range(rows):
          for j in range(cols):
              dst[i, j]=np.dot(T,src[i,j])
      return dst

  #Build Gaussian Pyramid
  def build_gaussian_pyramid(self, src,level=3):
      s=src.copy()
      pyramid=[s]
      for i in range(level):
          s=cv2.pyrDown(s)
          #print("Pyramid Shape: ", s.shape)
          pyramid.append(s)
      return pyramid

  #Build Laplacian Pyramid
  def build_laplacian_pyramid(self, src,levels=3):
      gaussianPyramid = self.build_gaussian_pyramid(src, levels)
      pyramid=[]
      for i in range(levels,0,-1):
          GE=cv2.pyrUp(gaussianPyramid[i])
          L=cv2.subtract(gaussianPyramid[i-1], GE)
          pyramid.append(L)
      return pyramid


  #load video from file
  def load_video(self, video_filename, mask_filename = None):
      cap=cv2.VideoCapture(video_filename)
      if mask_filename:
        cap_mask = cv2.VideoCapture(mask_filename)

      frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      # print("Video Resolution: ", frame_count, height, width)
      fps = int(cap.get(cv2.CAP_PROP_FPS))

      if not width % 8 == 0:
        r = width % 8
        if r >= 4:
          new_width = width+8-r
        else:
          new_width = width-r
      else:
        new_width = width
      
      if not height % 8 == 0:
        r = height % 8
        if r >= 4:
          new_height = height+8-r
        else:
          new_height = height-r
      else:
        new_height = height

      #print(new_height, new_width)
      if new_height < new_width:
        if new_width > 400:
          new_width = 400
          new_height = 280
      
      else:
        if new_height > 400:
          new_height = 400
          new_width = 280
        
      video_tensor = np.zeros((frame_count, new_height, new_width,3), dtype=np.uint8)
      mask_tensor = np.zeros((frame_count, new_height, new_width,3), dtype=np.bool)
      x=0
      while cap.isOpened():
          ret,frame=cap.read()
          if not mask_filename is None:
            _, frame_mask=cap_mask.read()
          #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
          if ret is True:
              #video_tensor[x]=frame
              # fg_mask = fgbg.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
              # fg_frame = cv2.bitwise_and(frame, frame, mask = fg_mask)
              # cv2_imshow(fg_mask)
              #cv2_imshow(fg_frame)
              video_tensor[x] = cv2.resize(frame,(new_width,new_height),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
              if not mask_filename is None:
                mask_tensor[x] = cv2.resize(frame_mask,(new_width,new_height),fx=0,fy=0)
                #video_tensor[x] = np.multiply(video_tensor[x], mask_tensor[x])

              #video_tensor[x] = cv2.resize(frame,(320,240),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
              x+=1
          else:
              break
      print(video_tensor.shape)
      return video_tensor, fps, mask_tensor

  # apply temporal ideal bandpass filter to gaussian video
  def temporal_ideal_filter(self, tensor,low,high,fps,axis=0):
      fft=fftpack.fft(tensor,axis=axis)
      frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
      bound_low = (np.abs(frequencies - low)).argmin()
      bound_high = (np.abs(frequencies - high)).argmin()
      fft[:bound_low] = 0
      fft[bound_high:-bound_high] = 0
      fft[-bound_low:] = 0
      iff=fftpack.ifft(fft, axis=axis)
      return np.abs(iff)

  # build gaussian pyramid for video
  def gaussian_video(self, video_tensor,levels=3):
      for i in range(0,video_tensor.shape[0]):
          frame=video_tensor[i]
          pyr=self.build_gaussian_pyramid(frame,level=levels)
          gaussian_frame=pyr[-1]
          if i==0:
              vid_data=np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1],3))
          vid_data[i]=gaussian_frame
      return vid_data

  #amplify the video
  def amplify_video(self, gaussian_vid,amplification=50):
      return gaussian_vid*amplification

  #reconstract video from original video and gaussian video
  def reconstract_video(self, amp_video,origin_video,levels=3):
      final_video=np.zeros(origin_video.shape)
      print("origin video shape: ", origin_video.shape)
      print("final video shape: ", amp_video.shape)
      for i in range(0,amp_video.shape[0]):
          img = amp_video[i]
          #print("img shape 1: ", img.shape)
          for x in range(levels):
              img=cv2.pyrUp(img)
              #print("img shape(pyrup): ", img.shape)
          #img=img[:, :origin_video.shape[2], :]+origin_video[i]
          img=img+origin_video[i]
          final_video[i]=img
      return final_video

  #save video to files
  def save_video(self, video_tensor, name):
      fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
      [height,width]=video_tensor[0].shape[0:2]
      writer = cv2.VideoWriter(name, fourcc, 30, (width, height), 1)
      for i in range(0,video_tensor.shape[0]):
          writer.write(cv2.convertScaleAbs(video_tensor[i]))
      writer.release()

  #magnify color
  def magnify_color(self, video_name, mask, out_name,low,high, save = True,
                    skin_detect = False, inpaint_bg = True, levels=3, amplification=20):
      t,f, mask_tensor=self.load_video(video_name, mask)
      bg = self.inpaint_background(t, mask_tensor)
      t = np.multiply(t, mask_tensor)
      if skin_detect:
        t = self.detect_skin(t)
      gau_video=self.gaussian_video(t,levels=levels)
      filtered_tensor=self.temporal_ideal_filter(gau_video,low,high,f)
      amplified_video=self.amplify_video(filtered_tensor,amplification=amplification)
      final=self.reconstract_video(amplified_video,t,levels=3)
      if inpaint_bg:
        final = final + bg
      if save:
        print("saving magnified video")
        self.save_video(final, out_name)
      self.mag_video = final
      return final
#      hr = HeartRate(video_path = "baby_monitoring/trimmed/37.mp4")
#      hr.run()


  #build laplacian pyramid for video
  def laplacian_video(self, video_tensor,levels=3):
      tensor_list=[]
      for i in range(0,video_tensor.shape[0]):
          frame=video_tensor[i]
          pyr=self.build_laplacian_pyramid(frame,levels=levels)
          if i==0:
              for k in range(levels):
                  tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1],3)))
          for n in range(levels):
              tensor_list[n][i] = pyr[n]
      return tensor_list

  #butterworth bandpass filter
  def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
      omega = 0.5 * fs
      low = lowcut / omega
      high = highcut / omega
      b, a = signal.butter(order, [low, high], btype='band')
      y = signal.lfilter(b, a, data, axis=0)
      return y

  #reconstract video from laplacian pyramid
  def reconstract_from_tensorlist(self, filter_tensor_list,levels=3):
      final=np.zeros(filter_tensor_list[-1].shape)
      print("final video shape: ", final.shape)
      for i in range(filter_tensor_list[0].shape[0]):
          up = filter_tensor_list[0][i]
          for n in range(levels-1):
              #up=cv2.pyrUp(up)[:, :filter_tensor_list[n+1][i].shape[1], :]+filter_tensor_list[n + 1][i]#可以改为up=cv2.pyrUp(up)
              up=cv2.pyrUp(up)+filter_tensor_list[n + 1][i]
              print("")
          final[i]=up
      return final

  # def detect_skin(self, video_tensor):
  #   final_video = np.zeros(video_tensor.shape)
  #   for i in range(video_tensor.shape[0]):
  #     f = video_tensor[i].astype('uint8')
  #     detector = skinDetector(f)
  #     mask, output = detector.find_skin()
  #     print(final_video[i].shape, output.shape)
  #     final_video[i] = output
  #   return final_video

  def inpaint_background(self, video, mask):
    fg_mask = 1-mask
    bg = video * fg_mask
    return bg

  #manify motion
  def magnify_motion(self, video_name, mask, out_name, 
                     low, high, save = True, skin_detect = False, inpaint_bg = True,
                     levels=3, amplification=20):
    
      t, f, mask_tensor=self.load_video(video_name, mask_filename = mask)
      bg = self.inpaint_background(t, mask_tensor)
      t = np.multiply(t, mask_tensor)
      #cv2_imshow(t[0])
      if skin_detect:
        t = self.detect_skin(t)
      lap_video_list=self.laplacian_video(t,levels=levels)
      filter_tensor_list=[]
      for i in range(levels):
          filter_tensor=self.butter_bandpass_filter(lap_video_list[i],low,high,f)
          filter_tensor*=amplification
          filter_tensor_list.append(filter_tensor)
      recon=self.reconstract_from_tensorlist(filter_tensor_list)
      final = t + recon
      if inpaint_bg:
        final = final + bg
      self.mag_video = final
      if save:
        self.save_video(final, out_name)
