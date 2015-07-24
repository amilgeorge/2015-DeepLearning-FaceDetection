# Guide on how to create training samples from WebFaces database using OpenCv utility.

1. Load the dataset [here](www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/).
2. Place it in the `Caltech_WebFaces` folder in `data`.
3. Run `extract_faces_alighned_webfaces.py`. It will crop faces and align them so that
   the eye's line is horizontal.
4. Run the `extract_non_faces_aflw.py` file to get negative examples. They are taken
   from the images where the faces are also present. But faces are not taken.
5. Run the bash command `find ./positive_images -iname "*.jpg" > positives.txt` to get a list of
   positive examples.
6. Same for the negative `find ./negative_images -iname "*.jpg" > negatives.txt`.
7. Run the `createtrainsamples.pl` file like this 
   `perl createtrainsamples.pl positives.txt negatives.txt vec_storage_tmp_dir`. Internally
   it uses `opencv_createsamples`. So you have to have it compiled. It will create a lot of
   `.vec` files in the specified directory.
8. Run `python mergevec.py -v vec_storage_tmp_dir -o final.vec`. You will have one `.vec` file
   with all the images.
9. Run the `vec2images final.vec output/%07d.png -w size -h size`. All the images will be in
   the output folder.