# -*- coding: utf-8 -*-
"""
NIFTI is a format for medical images.
NIFTI image is a 3d image of an organ it has three dimensions that are related to the 
three view we can have :1)axial 2)coronal 3)saggital.
We don't have patients data in NIFTI just we have some information about image in header
We use nibabel library for processing NIFTI files.
The other formt for medical images is DICOM it contains differnet slices of organs but because
working with NIFTI is easier we convert DICOM files to NIFTI files(one file contains all of slides)
In this code we try to load DICOM files convert them to NIFIT, show them with matplotlib
and save anrray as a NIFTI file
A nibabel image is the association of three things:
1:The image data array: a 3D or 4D array of image data
2:An affine array that tells you the position of the image array data in a reference space.
3:image metadata (data about the data) describing the image, usually in the form of an image header.
"""
import dicom2nifti
import os

os.chdir(r'C:\Users\Asus\Documents\py-torch\AI-IN-MEDICAL-MATERIALS\03-Data-Formats')
path_to_dicom =r"SE000001"
#single dot means that save new file in the same directory where SE000001 exists
dicom2nifti.convert_directory(path_to_dicom, ".")

'''
Read NIFTI files:
import the necessary packages:
1)nibabel to handle nifti files
2)matplotlib to plot the brain images
'''

import nibabel as nib
import matplotlib.pyplot as plt

#load nifti file
nifti = nib.load('201_t2w_tse.nii.gz')

#can access single metadata entries as follows:
print(nifti.header["qoffset_x"])

# get the image shape
print(nifti.shape)

#The image pixel data can be extracted using the get_fdata() function of the nifti object.
image_array = nifti.get_fdata()
print(image_array.dtype, image_array.shape)

#We can finally take a look at the brain scan.
#not forget to pass cmap="gray" to imshow, otherwise image will look quite odd

fig, axis = plt.subplots(5, 6, figsize=(10, 10))

slice_counter = 0
for i in range(5):
    for j in range(6):
        axis[i][j].imshow(image_array[:,:,slice_counter], cmap="gray")
        slice_counter+=1
        
'''
Write NIfTI files:
Manytimes, you will obtain image data as the results of an algorithm or processing
step that you want to store in the NIfTI format.
'''
# Here we apply a very simple threshold and set all image voxels to 0 that have
#a value smaller than 300
image_array_processed = image_array * (image_array>300)

# Now let us look at the results of this processing step (we just plot slice number 13) 
plt.imshow(image_array[:,:,13],cmap="gray") # plot the original image
plt.axis("off")

plt.imshow(image_array_processed[:,:,13],cmap="gray") # plot the processed image
plt.axis("off")

#First we convert the processed image array back to a nifti object.
processed_nifti = nib.Nifti1Image(image_array_processed, nifti.affine)

nib.save(processed_nifti, '201_t2w_tse_processed.nii.gz')
