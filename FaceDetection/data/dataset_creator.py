'''
Created on 04-Jul-2015

@author: amilgeorge
'''
from skimage.io._io import imread_collection
import shutil
from os import listdir
from os.path import isfile, join
import os
import sys

def execute():
    
    #faces_dir_path = r"/Users/amilgeorge/Documents/StudySS2015/DeepLearning/Training Data/data/raw_images/train_faces/13/"
    #bkgs_dir_path = r"/Users/amilgeorge/Documents/StudySS2015/DeepLearning/Training Data/cifar"
    #target_path = "/Users/amilgeorge/Documents/StudySS2015/DeepLearning/13"

    faces_dir_path = "data/processed_images/28_28_final_big_web/"
    bkgs_dir_path = "data/processed_images/28_28_nonfaces_aflw/"
    target_path = "data/processed_images/28_28_big_new/"

    train_dir = join(target_path, "train")
    validation_dir = join(target_path, "validation")

    num_train_faces = 40000
    num_train_bkgs = 18000
    
    num_validate_faces = 10000
    num_validate_bkgs = 4000

    img_faces = [ f for f in listdir(faces_dir_path) if isfile(join(faces_dir_path,f)) and f.endswith("png") ]
    img_bkgs =  [ f for f in listdir(bkgs_dir_path) if isfile(join(bkgs_dir_path,f)) and f.endswith("jpg") ]

    num_faces = len(img_faces)
    num_bkgs = len(img_bkgs)
    
    if num_faces < num_train_faces + num_validate_faces:
        print "I dont have enough faces"
        sys.exit(0)
    
    if num_bkgs < num_train_bkgs + num_validate_bkgs:
        print "I dont have enough faces"
        sys.exit(0)

    faces_subfolder_name = "faces"
    non_faces_subfolder_name = "nonfaces"

    # Creating all the necessary folders.
    if not os.path.exists(join(train_dir, faces_subfolder_name)):
        os.makedirs(join(train_dir, faces_subfolder_name))

    if not os.path.exists(join(train_dir, non_faces_subfolder_name)):
        os.makedirs(join(train_dir, non_faces_subfolder_name))

    if not os.path.exists(join(validation_dir, faces_subfolder_name)):
        os.makedirs(join(validation_dir, faces_subfolder_name))

    if not os.path.exists(join(validation_dir, non_faces_subfolder_name)):
        os.makedirs(join(validation_dir, non_faces_subfolder_name))
    
    for i, img_name in enumerate(img_faces):
        
        if i < num_train_faces:
            face_path = join(faces_dir_path,img_name)
            target_face_path = join(train_dir,faces_subfolder_name,img_name)
            shutil.copy(face_path,target_face_path)
        
        elif i < num_train_faces + num_validate_faces :
            face_path = join(faces_dir_path,img_name)
            target_face_path = join(validation_dir,faces_subfolder_name,img_name)
            shutil.copy(face_path,target_face_path)
        elif i >= num_train_faces + num_validate_faces:
            print "I'm done with faces..."
            break
    
    for i,img_name in enumerate(img_bkgs):
       
        if i < num_train_bkgs:
            bkgs_path = join(bkgs_dir_path,img_name)
            target_bkgs_path = join(train_dir,non_faces_subfolder_name,img_name)
            shutil.copy(bkgs_path,target_bkgs_path)
            
        elif i < num_train_bkgs + num_validate_bkgs :
            bkgs_path = join(bkgs_dir_path,img_name)
            target_bkgs_path = join(validation_dir,non_faces_subfolder_name,img_name)
            shutil.copy(bkgs_path,target_bkgs_path)
                        
        elif i >= num_train_bkgs + num_validate_bkgs :
            print "I'm done with bkgs..."
            break
        
        
    print("Work Done")
    

if __name__ == '__main__':
    execute()