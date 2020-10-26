import numpy as np
from .matlab_cp2tform import get_similarity_transform_for_cv2
import cv2
import random
from PIL import Image
import dlib

face_template = np.load('drive/My Drive/for_sphereface/face_template.npy')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('drive/My Drive/for_sphereface/shape_predictor_68_face_landmarks.dat')

BEARD = [5,6,7,8,9,10,11,12]
INNER_EYES_AND_BOTTOM_LIP = [38, 44, 30, 49, 54]



def alignment(src_img,src_pts):
  ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]

            
  crop_size = (96, 112)
  src_pts = np.array(src_pts).reshape(5,2)

  s = np.array(src_pts).astype(np.float32)
  r = np.array(ref_pts).astype(np.float32)

  tfm = get_similarity_transform_for_cv2(s, r)
  face_img = cv2.warpAffine(src_img, tfm, crop_size)
  return face_img

def beard_transformation(img, landmarks): #all landmarks in pairs

 #BEARD = [5,6,7,8,9,10,11,12]
 height = img.shape[0]
 beard_size = random.randint(int(height * 0.015), int(height * 0.025))
 beard_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

 for point in landmarks[BEARD]:
    cv2.circle(img, (point[0], point[1] -5), beard_size, beard_color, -1)

 return img

def circle_glasses_transformation(img, landmarks):
    
  #INNER_EYES_AND_BOTTOM_LIP = [38, 44, 30, 49, 54]
  landmarks = landmarks[INNER_EYES_AND_BOTTOM_LIP]
  height = img.shape[0]
  eyes_size = random.randint(int(height * 0.025), int(height * 0.035))
  img = cv2.circle(img, tuple(landmarks[0]), eyes_size,
                    (0, 0, 0), -1)
  img = cv2.circle(img, tuple(landmarks[1]), eyes_size,
                    (0, 0, 0), -1)
  return img 


def rect_glasses_transformation(img, landmarks):
    
  #INNER_EYES_AND_BOTTOM_LIP = [38, 44, 30, 49, 54]
  landmarks = landmarks[INNER_EYES_AND_BOTTOM_LIP]  
  height = img.shape[0]
  eyes_size = random.randint(int(height * 0.025), int(height * 0.035))
  
  img = cv2.rectangle(img, tuple((landmarks[0][0] - eyes_size, landmarks[0][1] + eyes_size)),
           tuple((landmarks[0][0] + eyes_size, landmarks[0][1] - eyes_size)),
                    (0, 0, 0), -1)
  img = cv2.rectangle(img, tuple((landmarks[1][0] - eyes_size, landmarks[1][1] + eyes_size)),
           tuple((landmarks[1][0] + eyes_size, landmarks[1][1] - eyes_size)),
                    (0, 0, 0), -1)
  return img


def cut_transformation(img, landmarks):
    
  #INNER_EYES_AND_BOTTOM_LIP = [38, 44, 30, 49, 54]
  landmarks = landmarks[INNER_EYES_AND_BOTTOM_LIP]
  height = img.shape[0]
  rand = random.randint(int(height*0.05),int(height*0.15))
  img = img[(landmarks[0][1] - rand):(landmarks[4][1] + rand), (landmarks[0][0] - rand):(landmarks[4][0] + rand)]
  
  return img  


def rotate_transformation(img):

 (h, w, d) = img.shape
 center = (w // 2, h // 2)
 angle = random.randint(0,45)
 M = cv2.getRotationMatrix2D(center, angle, 1.0)
 img = cv2.warpAffine(img, M, (w, h))

 return img  



def reflect(img):
 #img = Image.open(img)
 img = img.transpose(Image.FLIP_LEFT_RIGHT)
 img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
 return img  


def alignment_bias(src_img,src_pts,x,y):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        x,y,[62.7299, 92.2041] ]

            
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

def transformation(img, landmarks):
  #INNER_EYES_AND_BOTTOM_LIP = [38, 44, 30, 49, 54]  
  landmarks = landmarks[INNER_EYES_AND_BOTTOM_LIP]
  param1 = random.randint(0,15)
  param2 = random.randint(0,15)
  img = alignment_bias(img, landmarks, [60 - param1, 80 - param2],[40 - param1, 100 - param2])
  

  return img


def face_detection(img):
 
 face_rects = list(detector(img, 1))  
    
 return face_rects



def landmarks_extracting(img, face_rect):
    
 points = predictor(img, face_rect)
 landmarks = np.array([*map(lambda p: [p.x, p.y], points.parts())])

 return landmarks

