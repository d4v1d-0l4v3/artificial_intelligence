# Author: David O
# Date: July 2024
# Description: Loading of diabetic-retinopathy images following directory
#              structure based on disease advanced stages (0 => No disease to
#              4 => Adavance stage)

# Python imports
import pandas as pd
from operator import itemgetter
import numpy as np
import shutil

# definitions
base_dir = "/media/davidolave/OS/Downloads/ai_datasets/diabetic-retinopathy-detection"
# Flattened Directory containing unorganized trainining images
flat_dir_training_imgs = base_dir + "/training_images"
# Note: Level/label 0 => No identified disease. Level/label 4 => Most advance identified disease stage
labels_file_path = base_dir + "/trainLabels.csv"
# Top directory that contains images organized on label (disease stage) based directories
top_dir_structured_labeled_images_dir = base_dir + "/disease_stages_images"
# Directory containing no disease images
labeled0_file_path = top_dir_structured_labeled_images_dir + "/level_0"
# Directory containing images with the most advance state of disease
labeled4_file_path = top_dir_structured_labeled_images_dir + "/level_4"

# Process csv file with filename to disease-stage mappings
df = pd.read_csv(labels_file_path)
labels_list_to_sort = [(file_name, level) for (file_name, level) in df.itertuples(index=False)]
# sorted_labels_list = sorted(labels_list_to_sort, key=itemgetter(0))  # Sort by filename
# sorted_labels_list = [item[1] for item in sorted_labels_list]
# Max number images associated with most advance stage of disease in training dataset
# max_num_level4_train_imgs = 638
max_num_level4_train_imgs = 20000

# Organize image files in directories based on stage level. Currently, only organized for
# level 0 (No dr disease found) and level 4 (classified as highest stage of disease in dataset)
no_dr_ctr = 0
level4_dr_ctr = 0
total_training_images = 0
img_ext = ".jpeg"
for file_name, level in labels_list_to_sort:
    copy_file = False  # Do not copy file yet
    total_training_images = total_training_images + 1

    src_file_path = flat_dir_training_imgs + "/" + file_name + img_ext
    dst_file_path = ""
    # Needs to limit the number of no disease images since a number imbalance exist between 'no dr'
    # and number of images with most advance stage of disease
    if (level == 0) & (no_dr_ctr < max_num_level4_train_imgs):  # No disease and limit number of images to include
        no_dr_ctr = no_dr_ctr + 1
        copy_file = True
        dst_file_path = labeled0_file_path + "/" + file_name + img_ext
    elif level == 4:
        level4_dr_ctr = level4_dr_ctr + 1
        copy_file = True
        dst_file_path = labeled4_file_path + "/" + file_name + img_ext

    if copy_file:
        # Move from unorganized to label-based structure directory
        shutil.copyfile(src_file_path, dst_file_path)

print ("no_dr_ctr=", no_dr_ctr, " level4_dr_ctr=", level4_dr_ctr)
