import SimpleITK as sitk
import numpy as np
import h5py

subject_list = [253, 255, 257, 258, 260, 262, 263, 264, 265, 266, 267, 269, 270, 275, 283, 284, 286, 287, 288, 289]

mandible = []
midface = []
for idx in range(len(subject_list)):
    subject_id = subject_list[idx]
    print('Subject {0}'.format(subject_id))

    image = sitk.ReadImage('./segmentation/subject{0}/multiscale_subject_{0}.nii.gz'.format(subject_id))
    image = sitk.GetArrayFromImage(image)

    label0 = np.zeros_like(image, dtype=np.float32)
    label0[np.where(image == 0)] = 1.

    label1 = np.zeros_like(image, dtype=np.float32)
    label1[np.where(image == 1)] = 1.

    label2 = np.zeros_like(image, dtype=np.float32)
    label2[np.where(image == 2)] = 1.

    
    groundtruth = sitk.ReadImage('/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/label_nii/subject{0}.nii.gz'.format(subject_id))
    groundtruth = sitk.GetArrayFromImage(groundtruth)

    #a = 2.0 * np.sum((prediction == groundtruth).astype(np.int32))
    #b = np.sum(prediction) + np.sum(groundtruth)

    groundtruth1 = np.zeros_like(groundtruth, dtype=np.float32)
    groundtruth1[np.where(groundtruth == 1)] = 1.
    groundtruth1 = groundtruth1.astype(np.bool)

    intersection = np.logical_and(label1, groundtruth1)
    im_sum = label1.sum() + groundtruth1.sum()

    print('Dice | mandible is: {}'.format(2. * intersection.sum() / im_sum))
    mandible.append(2. * intersection.sum() / im_sum)

    groundtruth2 = np.zeros_like(groundtruth, dtype=np.float32)
    groundtruth2[np.where(groundtruth == 2)] = 1.
    groundtruth2 = groundtruth2.astype(np.bool)

    intersection = np.logical_and(label2, groundtruth2)
    im_sum = label2.sum() + groundtruth2.sum()

    print('Dice | midface is: {}'.format(2. * intersection.sum() / im_sum))
    midface.append(2. * intersection.sum() / im_sum)
    print('\n')

print('mandible: ', mandible)
print(np.mean(mandible), np.std(mandible))

print('midface: ', midface)
print(np.mean(midface), np.std(midface))
