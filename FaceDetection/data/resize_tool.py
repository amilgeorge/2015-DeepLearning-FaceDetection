'''
Created on 31-Jul-2015

@author: amilgeorge
'''

from os import listdir
from os.path import isfile, join
import os
from skimage.io._io import imread, imsave
from skimage.transform._warps import resize
from skimage.color.colorconv import rgb2gray
from skimage.util.dtype import img_as_ubyte



def execute(outputsize):
    
    faces_dir_path = "data/train_set/48_48_faces_web_augmented"
    bkgs_dir_path = "data/train_set/48_48_nonfaces_aflw"
    
    target_path = "data/train_set/13"
    
    
    faces_dir=join(target_path,"faces")
    nonfaces_dir = join(target_path,"nonfaces") 
    
    
 
    os.makedirs(nonfaces_dir)
    os.makedirs(faces_dir)
    
    img_faces = [ f for f in listdir(faces_dir_path) if isfile(join(faces_dir_path,f)) and f.endswith("png") ]
    img_bkgs =  [ f for f in listdir(bkgs_dir_path) if isfile(join(bkgs_dir_path,f)) and f.endswith("jpg") ]
    
    for i, img_name in enumerate(img_faces):
        img_path = join(faces_dir_path,img_name)
        img = imread(img_path)
        resized_img = resize(img,outputsize)     
        ubyte_img = img_as_ubyte(resized_img)   
        imsave(join(faces_dir,img_name), ubyte_img)
        print "processed "+ img_path
        
    for i, img_name in enumerate(img_bkgs):
        img_path = join(bkgs_dir_path,img_name)
        img = imread(img_path)
        gray_img = rgb2gray(img)  
        resized_img = resize(gray_img,outputsize)    
        ubyte_img = img_as_ubyte(resized_img)            
        imsave(join(nonfaces_dir,img_name), ubyte_img)
        print "processed "+ img_path
        
def convert_to_gray():
    bkgs_dir_path = "data/train_set/13/nonfaces"
    target_path = "data/train_set/13/nonfaces_gray"
     
    os.makedirs(target_path)
    img_bkgs =  [ f for f in listdir(bkgs_dir_path) if isfile(join(bkgs_dir_path,f)) and f.endswith("jpg") ]
     
     
    for i, img_name in enumerate(img_bkgs):
        img_path = join(bkgs_dir_path,img_name)
        img = imread(img_path)
        gray_img = rgb2gray(img)
        imsave(join(target_path,img_name), gray_img)
        
        
if __name__ == '__main__':
    execute((13,13))
    #convert_to_gray()