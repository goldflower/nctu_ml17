import numpy as np
import cv2, os, sys
import scipy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.style.use('ggplot')
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import sys
sys.path.insert(0, 'FaceDetect-master')
import multiprocessing
import pandas as pd

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
age = ['adult', 'child', 'elder', 'young']
sexual = ['male', 'female']

def get_faces(data_path='data/', save_data_path='processed', age = ['adult', 'child', 'elder', 'young'], 
              sexual = ['male', 'female'], preserve_all=True, min_neighbor=3, second_detector=False,
              scale = None, write_origin=True):
    if second_detector:
        import FaceFinder
        import tensorflow as tf
        model_path = 'FaceDetect-master/face_model'
        def tf_detector_helper(file_path):
            tf.reset_default_graph()
            img = cv2.imread(path + file, 0)
            tf_faces, mask = FaceFinder.localize(img, model_path)
            return tf_faces, mask
    if scale:
        def get_scaling_faces(img, scale):
            i = cv2.resize(img, (int(img.shape[0]*scale), img.shape[1]))
            g = cv2.resize(gray, (int(gray.shape[0]*scale), gray.shape[1]))
            faces = face_cascade.detectMultiScale(g, scaleFactor=1.3, minNeighbors=min_neighbor)
            
            if len(faces) == 0:
                i = cv2.resize(img, (img.shape[0], int(img.shape[1]*scale)))
                g = cv2.resize(gray, (gray.shape[0], int(gray.shape[1]*scale)))
                faces = face_cascade.detectMultiScale(g, scaleFactor=1.3, minNeighbors=min_neighbor)
                
            return faces, i, g
    
    for a in age:
        for s in sexual:
            print(a, s, min_neighbor)
            path = data_path + a + '/' + s + '/'
            save_path = save_data_path + str(min_neighbor) + '/' + a + '/' + s + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            files = os.listdir(path)
            chunk = len(files) // 80
            for file, count in zip(files, range(len(files))):
                try_rescale = False
                if count % chunk == 0:
                    sys.stdout.write("\r[%s%s]" % ('=' * (count//chunk), ' ' * (80-count//chunk)))
                img = cv2.imread(path + file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=min_neighbor)
                cv2.imwrite((save_path + file).replace('.','_o.'), img)
                if scale is None and len(faces) == 0:
                    if perserve_all:
                        csv.imwrite(save_path + file, img)
                    continue
                elif scale and len(faces) == 0: # hass fail for origin image
                    try_rescale = True
                    faces, img, gray = get_scaling_faces(gray, scale)
                    
                
                has_face = False
                processed_img = None
                for (row, col, height, width), i in zip(faces, range(len(faces))):
                    if height < min(img.shape[0], img.shape[1])*0.2:
                        if i == len(faces)-1: # hass fail
                            break
                        else:                     
                            continue
                    elif processed_img is None:                 
                        processed_img = img[col:col+width, row:row+height]
                    elif height > processed_img.shape[0]: # keep the largest square              
                        processed_img = img[col:col+width, row:row+height]
                else:
                    if processed_img is None:
                        continue
                    cv2.imwrite(save_path + file, processed_img)
            print()


get_faces(save_data_path='rescale_preserve_origin', scale=1.2)