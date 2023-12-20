import os, shutil

import numpy as np
import nibabel as nib

from icad.preprocessing.utils import registration, registration_mask, mat_to_affine

def preprocess_case(case_id, atlas_path, cta_path, ncct_path, vessel_segmentation_path, thrombus_segmentation_path):
    """
    Preprocesses a case by registering the CTA and NCCT to the MNI atlas 
    in the median spacing of the dataset and applying the transformation 
    to the vessel and thrombus segmentations.

    Parameters
    ----------
    case_id : str
        Case ID.
    atlas_path : str
        Path to the MNI atlas.
    cta_path : str
        Path to the CTA image.
    ncct_path : str
        Path to the NCCT image.
    vessel_segmentation_path : str
        Path to the vessel segmentation.
    thrombus_segmentation_path : str
        Path to the thrombus segmentation.
    
    """
    # Define paths
    images_db = os.path.join(os.environ["icad_base_path"], "database/preprocessed_icad/images")
    labels_db = os.path.join(os.environ["icad_base_path"], "database/preprocessed_icad/labels")
    transforms_db = os.path.join(os.environ["icad_base_path"], "database/preprocessed_icad/registration_transforms")
    # Define output paths
    output_ncct_path = os.path.join(images_db, f"{case_id}_0000.nii.gz")
    output_cta_path = os.path.join(images_db, f"{case_id}_0001.nii.gz")
    output_vessels_path = os.path.join(labels_db, f"{case_id}_0000.nii.gz")
    output_thrombus_path = os.path.join(labels_db, f"{case_id}_0001.nii.gz")

    print(f"Preprocessing case {case_id}...")

    # Register CTA to MNI atlas in median spacing
    print("Registering CTA to MNI space...")
    transform_path = registration(  fixed_image_path=atlas_path, 
                                    moving_image_path=cta_path, 
                                    output_image_path=output_cta_path, 
                                    transformation_type="Affine", 
                                    save_transform=True)
    # Convert the mat file to affine matrix and save it
    with open(transform_path, "r") as f:
        lines = f.readlines()
        line = [line.strip() for line in lines][0]

    # Copy the file to the present directory
    shutil.copy(line, os.path.join(transforms_db, f"{case_id}_0001.mat"))
    # Create affine matrix
    affine = mat_to_affine(os.path.join(transforms_db, f"{case_id}_0001.mat"))
    # Save affine matrix
    np.save(os.path.join(transforms_db, f"{case_id}_0001.npy"), affine)
    # Remove the txt file
    os.remove(transform_path)

    # Register NCCT to registered CTA
    print("Registering NCCT to MNI space...")
    transform_path = registration(  fixed_image_path=output_cta_path, 
                                    moving_image_path=ncct_path, 
                                    output_image_path=output_ncct_path, 
                                    transformation_type="Affine", 
                                    save_transform=True)
    # Convert the mat file to affine matrix and save it
    with open(transform_path, "r") as f:
        lines = f.readlines()
        line = [line.strip() for line in lines][0]

    # Copy the file to the present directory
    shutil.copy(line, os.path.join(transforms_db, f"{case_id}_0000.mat"))
    # Create affine matrix
    affine = mat_to_affine(os.path.join(transforms_db, f"{case_id}_0000.mat"))
    # Save affine matrix
    np.save(os.path.join(transforms_db, f"{case_id}_0000.npy"), affine)
    # Remove the txt file
    os.remove(transform_path)
    # Define the path to the CTA transformation .mat file
    transformation_file = os.path.join(transforms_db, f"{case_id}_0001.mat")

    # Apply the CTA transformation to the vessel segmentation
    print("Registering vessel segmentation to MNI space...")
    registration_mask(  fixed_image_path=atlas_path,
                        moving_mask_path=vessel_segmentation_path,
                        output_mask_path=output_vessels_path,
                        transformation_file=transformation_file,
                        remove_transformation=False)

    # Apply the CTA transformation to the thrombus segmentation
    print("Registering thrombus segmentation to MNI space...")
    registration_mask(  fixed_image_path=atlas_path,
                        moving_mask_path=thrombus_segmentation_path,
                        output_mask_path=output_thrombus_path,
                        transformation_file=transformation_file,
                        remove_transformation=False)
    
    print("Preprocessing finished\n")