
### loading and preprocesing Data with torchio
# 
# We make two folders. One folder contains the nifti images for each patient and the other folder contains the labels od nifti images . They are imagestr and labelstr. 
# 
# For example if we have 300 patienst then we have 300 files in each folder. And both of them are in kits21 folder.

from pathlib import Path
dataset_dir = Path('/home/jacobo/Documents/MM-p/kits21/kidneys')


# We load the files and transfer them to the image_paths and labels_paths


images_dir = dataset_dir /'imagestr'
labels_dir = dataset_dir /'labelstr'
image_paths = sorted(images_dir.glob('*.nii.gz'))
label_paths = sorted(labels_dir.glob('*.nii.gz'))
assert len(image_paths) == len(label_paths)


# Torchio is an appropriate library for loading medical image data and for preprocessing
# This is the link to all of transformations we can use by torchio:
# 
# https://torchio.readthedocs.io/transforms/transforms.html.
# 
# We load all of images and labels and make a dataset with subjectsdatase


import torch
import torchio as tio
import torch.nn.functional as F
subjects = []
for (image_path, label_path) in zip(image_paths, label_paths):
    subject = tio.Subject(
        mri=tio.ScalarImage(image_path),
        label=tio.LabelMap(label_path),
    )
    subjects.append(subject)
dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')
print(dataset)


# Plot the first subject that contains images and labels

one_subject = dataset[0]
one_subject.plot()


print(one_subject)
print(one_subject.mri)
print(one_subject.label)


# We can use tio.compose for two purposes: preprocessing- augmentation
# 
# We did transform for both training and validation data


training_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(4),
    tio.CropOrPad((48, 60, 48)),
    tio.RandomMotion(p=0.2),
    tio.RandomBiasField(p=0.3),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.RandomNoise(p=0.5),
    tio.RandomFlip(),
    tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    }),
    tio.OneHot(),
])

validation_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(4),
    tio.CropOrPad((48, 60, 48)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.OneHot(),
])


# We want to seperate training and validation data that both of them are in image_paths


num_subjects = len(dataset)
training_split_ratio=0.9
num_training_subjects = int(training_split_ratio * num_subjects)
num_validation_subjects = num_subjects - num_training_subjects



num_split_subjects = num_training_subjects, num_validation_subjects
#print(num_split_subjects)
#print(subjects)
training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)


# Apply transforms to subjects



training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)
validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)




print('Training set:', len(training_set), 'subjects')
print('Validation set:', len(validation_set), 'subjects')


# Plot the the subject with transform



training_instance = training_set[0]  # transform is applied inside SubjectsDataset
training_instance.plot()



import multiprocessing
num_workers = multiprocessing.cpu_count()


# Make a train loader from subjects dataset for sending it to the training model



training_batch_size = 1
validation_batch_size = 1 * training_batch_size

training_loader = torch.utils.data.DataLoader(
    training_set,
    batch_size=training_batch_size,
    shuffle=True,
    num_workers=num_workers,
)

validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=validation_batch_size,
    num_workers=num_workers,
)

