# -*- coding: utf-8 -*-
"""
We try to review preprocessing for medical images in this file. we did these things 
in this code:
    1-understanding the view of images (axial,coronal,saggital). When we get the data 
    with nibabel and load it to array, we can understand every dimension of array belongs
    to which view
    2-understanding voxel sizes
    3-slicing the images in different views
    4-swapping and flipping images
    5-physical to voxel and voxel to physical transformation
    6-reorient the image to RAS
    7-resample a volume to a smaller size
    8-CT and MRI standardization
    9-Windowing in CT(change contrasts)

"""
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

#load data
brain_mri = nib.load("IXI662-Guys-1120-T1.nii.gz")
#send nifti to an array
brain_mri_data = brain_mri.get_fdata()

#Then we extract its shape and affine matrix
'''
The affine matrix describes the mapping from the pixel coordinates to the scanner
(or world) coordinates.
It is always a 4×4 matrix. The left 3×3 sub-matrix is responsible for rotation, scaling
and shearing the coordinates. The 4th column is responsible for the translation or the offset
'''
shape = brain_mri.shape
affine = brain_mri.affine
print(affine)
print(shape)

'''
Voxel is short for volume pixel, the smallest distinguishable box-shaped part of a 3D image 
We can access the size of a voxel in the volume by using the header.get_zooms() function
provided by nibabel
'''
print(brain_mri.header.get_zooms())

'''
Slicing:
You can slice through the volume in all orientations: axial, coronal and sagittal.
NOTE: Depending on the orientation of the scan, the indices change
Sometimes the first axis slices axial, sometimes coronal and sometimes sagittal.
You can find out the orientation by using nib.aff2axcodes(affine) 
'''
print(nib.aff2axcodes(affine))

'''
In this case, the orientation of the scan is:
from anterior to posterior (from front to back)
from inferior to superior (from bottom to top)
from left to right
The letters returned from aff2axcodes always indicate the end of the corresponding axis.
'''
################################################################################
#Coronal view: At first we move along the first axis (anterior to posterior).  #
#This moves through the head from the face to the back of the head. the picture#
#is upside down because it was anterior to posterio                            #
################################################################################
fig, axis = plt.subplots(1, 2)
axis[0].imshow(brain_mri_data[40, :, :], cmap="gray")
axis[1].imshow(brain_mri_data[120, :, :], cmap="gray")

#################################################################################
#Axial view: we slice along the second axis which moves from the lower jaw/ neck#
#to the top of the head.                                                        #
#################################################################################
fig, axis = plt.subplots(1, 2)
axis[0].imshow(brain_mri_data[:, 30, :], cmap="gray")
axis[1].imshow(brain_mri_data[:, 200, :], cmap="gray")

##################################################################################
#Sagital view: finally we slice through the third axis which moves from the right#
#ear to the left ear.                                                            #
#the head is rotated because we move from front to back                          #
##################################################################################
fig, axis = plt.subplots(1, 2)
axis[0].imshow(brain_mri_data[:, :, 20], cmap="gray")
axis[1].imshow(brain_mri_data[:, :, 45], cmap="gray")

#swapp axis in showing the picture.Interchange two axes of an array.
fig, axis = plt.subplots(1, 2)

brain_mri_swapped = np.swapaxes(brain_mri_data, 0, 1)
axis[0].imshow(brain_mri_swapped[:, :, 20], cmap="gray")
axis[1].imshow(brain_mri_swapped[:, :, 45], cmap="gray")

##################################################################################
#You can use np.flip(arr, axis) to flip the axis and thus changes the orientation#
#to top->bottom.                                                                 #
#CAUTION: This does not change the affine matrix and must only be used for       #
#validation purposes                                                             #
##################################################################################
fig, axis = plt.subplots(1, 2)

axis[0].imshow(np.flip(brain_mri_swapped, 0)[:, :, 20], cmap="gray")
axis[1].imshow(np.flip(brain_mri_swapped, 0)[:, :, 100], cmap="gray")

'''
transform coordinates between the coordinate systems
calculate the physical coordinates of the offset, i.e where the voxel coordinates
(0,0,0) lie in the physical space, you simply multiply the affine matrix with those
coordinates.
'''
#voxel to physical
voxel_coord = np.array((0, 0, 0, 1))
physical_coord0 = affine @ voxel_coord  # @ is a shortcut for matrix multiplication in numpy
print(physical_coord0)

#physical to voxel
'''
If you want to transform physical coordinates into pixel/voxel coordinates you
need to compute the inverse of the affine matrix (np.linalg.inv(arr) and then 
multiply this inverse with the physical coordinates
'''
voxel_coords = (np.linalg.inv(affine) @ physical_coord0).round()
print(voxel_coords)

'''
Reorientation:If you want, you can reorient the volume to RAS by using 
nibabel.as_closest_canonical(nifti)
This is also called canonical orientation
'''
brain_mri_canonical = nib.as_closest_canonical(brain_mri)
brain_mri_canonical_data = brain_mri_canonical.get_fdata()
canonical_affine = brain_mri_canonical.affine
print(nib.aff2axcodes(canonical_affine))

fig, axis = plt.subplots(1, 2)
axis[0].imshow(brain_mri_canonical_data[50, :, :], cmap="gray")
axis[1].imshow(brain_mri_canonical_data[130, :, :], cmap="gray")

fig, axis = plt.subplots(1, 2)
axis[0].imshow(brain_mri_canonical_data[:, 40, :], cmap="gray")
axis[1].imshow(brain_mri_canonical_data[:, 90, :], cmap="gray")

fig, axis = plt.subplots(1, 2)
axis[0].imshow(brain_mri_canonical_data[:, :, 5], cmap="gray")
axis[1].imshow(brain_mri_canonical_data[:, :, 70], cmap="gray")

'''
Resampling:
resample a volume to smaller size. change the size of your scan as it might be 
too large for your system. However, resizing a volume is not as easy as resizing
an image because the voxel size needs to be changed. 
Let's resize our brain mri from (256, 256, 150) to (128, 128, 100)
'''
print(brain_mri.shape)
print(brain_mri.header.get_zooms())

#use conform(input, desired_shape, voxel_size) from nibabel

import nibabel.processing
voxel_size = (2, 2, 2)
brain_mri_resized = nibabel.processing.conform(brain_mri, (128, 128, 100), voxel_size, orientation="PSR")
brain_mri_resized_data = brain_mri_resized.get_fdata()

print(brain_mri.shape)
print(brain_mri_resized.shape)
print(brain_mri_resized.header.get_zooms())

# the resampled image still look similar to the original! a little different in
#resolution and voxel visualization
IDX = 50
fig, axis = plt.subplots(1, 2)
axis[0].imshow(brain_mri_data[:,:,IDX], cmap="gray")
axis[1].imshow(brain_mri_resized_data[:,:,IDX], cmap="gray") 


'''
Standardization:
As CTs have a fixed scale from -1000 (air) to 1000 (water) you normally do not 
perform normalization to keep those scales.
In practice, you can assume that the values are between -1024 and 3071.
Thus you can standardize the data by dividing the volume by 3071.
'''
lung_ct = nib.load("lung_043.nii.gz")
lung_ct_data = lung_ct.get_fdata()
plt.figure()
#np.rot90 rotates the picture 90 degree
plt.imshow(np.rot90(lung_ct_data[:,:,50]), cmap="gray")

lung_ct_data_standardized = lung_ct_data / 3071
plt.figure()
plt.imshow(np.rot90(lung_ct_data_standardized[:,:,50]), cmap="gray")

'''
Windowing:
Depending on the task you perform you want to have a different contrast.
This change in contrast is called windowing.
There are typically four different windows, a lung window, a bone window, a
soft-tissue window and a brain window.
You can create such a window, by clipping all pixel values larger than a threshold.
'''
#lung window. Note that this window completely denies us to take a look at the 
#abdomen as everything looks identical. The thereshold is between -1000 and 500

lung_ct_lung_window = np.clip(lung_ct_data, -1000, -500)

fig, axis = plt.subplots(1, 2)
axis[0].imshow(np.rot90(lung_ct_lung_window[:,:,50]), cmap="gray")
axis[1].imshow(np.rot90(lung_ct_lung_window[:,:,5]), cmap="gray")
axis[0].axis("off")
axis[1].axis("off")
fig.suptitle("Lung Window")
plt.tight_layout()
plt.savefig("lung_window.png", bbox_inches="tight")

#soft-tissue window. Here the lung ist almost black but you have a nice image
#of the abdomen. The thereshold is between -250 and 250

lung_ct_soft_tissue_window = np.clip(lung_ct_data, -250, 250)

fig, axis = plt.subplots(1, 2)
axis[0].imshow(np.rot90(lung_ct_soft_tissue_window[:,:,50]), cmap="gray")
axis[1].imshow(np.rot90(lung_ct_soft_tissue_window[:,:,5]), cmap="gray")

axis[0].axis("off")
axis[1].axis("off")
fig.suptitle("Soft Tissue Window")
plt.tight_layout()
plt.savefig("tissue_window.png", bbox_inches="tight")

'''
Standardization:
In contrast to CTs MRI images do not have an absolute, fixed scale and each 
patient varies.
'''
cardiac_mri = nib.load("la_003.nii.gz")
cardiac_mri_data = cardiac_mri.get_fdata()

mean, std = np.mean(cardiac_mri_data), np.std(cardiac_mri_data)
cardiac_mri_norm = (cardiac_mri_data - mean) / std
cardiac_mri_standardized = (cardiac_mri_norm - np.min(cardiac_mri_norm)) / (np.max(cardiac_mri_norm) - np.min(cardiac_mri_norm))

plt.figure()
plt.imshow(cardiac_mri_standardized[:,:,30], cmap="gray")

'''
Typically there is no windowing in MRI scans.
'''