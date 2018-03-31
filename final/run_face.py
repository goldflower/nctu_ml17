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


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def get_faces(data_path='data/', save_data_path='processed', age = ['adult', 'child', 'elder', 'young'], 
              sexual = ['male', 'female'], preserve_all=False, min_neighbor=3, second_detector=True):
    if second_detector:
        import FaceFinder
        import tensorflow as tf
        model_path = 'FaceDetect-master/face_model'
        def tf_detector_helper(file_path):
            tf.reset_default_graph()
            img = cv2.imread(path + file, 0)
            tf_faces, mask = FaceFinder.localize(img, model_path)
            return tf_faces, mask
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
                if count % chunk == 0:
                    sys.stdout.write("\r[%s%s]" % ('=' * (count//chunk), ' ' * (80-count//chunk)))
                img = cv2.imread(path + file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=min_neighbor)
                if len(faces) == 0: # hass fail
                    if second_detector: # use tensorflow
                        tf_faces, mask = tf_detector_helper(path+file)
                        if tf_faces is None: # tf fail
                            if preserve_all:
                                cv2.imwrite(save_path + file, img)
                        else:
                            cv2.imwrite(save_path + file, tf_faces)
                    elif not second_detector and perserve_all:
                        csv.imwrite(save_path + file, img)
                    continue
                has_face = False
                processed_img = None
                for (row, col, height, width), i in zip(faces, range(len(faces))):
                    if height < min(img.shape[0], img.shape[1])*0.2:
                        if i == len(faces)-1: # hass fail
                            if second_detector: # use tensorflow
                                tf_faces, mask = tf_detector_helper(path+file)
                                if tf_faces is None: # tf fail
                                    if preserve_all:
                                        cv2.imwrite(save_path + file, img)
                                else:
                                    cv2.imwrite(save_path + file, tf_faces)
                            elif not second_detector and preserve_all:
                                cv2.imwrite(save_path + file, img)                            
                            break
                        else:                     
                            continue
                    elif processed_img is None:                 
                        processed_img = img[col:col+width, row:row+height]
                    elif height > processed_img.shape[0]: # keep the largest square              
                        processed_img = img[col:col+width, row:row+height]
                else:
                    cv2.imwrite(save_path + file, processed_img)
            print()

if __name__ == '__main__':
    p1 = multiprocessing.Process(target = get_faces, 
                                 args = ('data/', 'processed_opencv_then_tf', ['adult', 'child'], 
                                         ['male', 'female'], False, 3, True,))
    p2 = multiprocessing.Process(target = get_faces, 
                                 args = ('data/', 'processed_opencv_then_tf', ['elder', 'young'], 
                                         ['male', 'female'], False, 3, True,))
    p1.start()
    p2.start()

    print("The number of CPU is: " + str(multiprocessing.cpu_count()))
    for p in multiprocessing.active_children():
        print("child   p.name:" + p.name + "\tp.id " + str(p.pid))

    # p1.join()
    # p2.join()            