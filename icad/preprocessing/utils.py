import os
import ants

import numpy as np
import nibabel as nib

import scipy.io as sio

from scipy.ndimage import zoom

def registration(fixed_image_path, moving_image_path, output_image_path, transformation_type = "Affine", save_transform = False, label = False):
    """
    Performs registration of the moving image to the fixed image. Possible types of transform: "AffineFast", "Affine", "SyN"

    Parameters
    ----------
    fixed_image_path : str
        Path to the fixed image.
    moving_image_path : str
        Path to the moving image.
    output_image_path : str
        Path to the output image.
    transformation_type : str, optional
        Type of transformation. The default is "Affine".
    save_transform : bool, optional
        If True, the transformation matrix is saved. The default is False.
    label : bool, optional
        If True, the transformation is applied to the moving image using genericLabel interpolator. The default is False.

    Raises
    ------
    TypeError
        If the transformation type is not one of the following: "AffineFast", "Affine", "SyN".
    """
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)

    if transformation_type == "AffineFast":
        transform = ants.registration(fixed_image, moving_image, type_of_transform = 'AffineFast')
    elif transformation_type == "Affine":
        transform = ants.registration(fixed_image, moving_image, type_of_transform = 'Affine')
    elif transformation_type == "SyN":
        transform = ants.registration(fixed_image, moving_image, type_of_transform = 'SyN')
    
    # Apply the transformation to the moviComputeimage
    if label:
        warped_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=transform['fwdtransforms'], interpolator='genericLabel')
    else:
        warped_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=transform['fwdtransforms'])
    
    # Save the transformed image
    warped_image.to_file(output_image_path)

    # Pass dtype of final nifti to int16
    nifti = nib.load(output_image_path)
    nib.save(nifti, output_image_path, dtype = np.int32)
    
    if save_transform:
        # Save the transformation matrix
        np.savetxt(os.path.join(os.path.dirname(output_image_path), 'transfomation_matrix.txt'), transform['fwdtransforms'], fmt='%s')
        return os.path.join(os.path.dirname(output_image_path), 'transfomation_matrix.txt')
        
def registration_mask(fixed_image_path, moving_mask_path, output_mask_path, transformation_file = None, remove_transformation = True):
    """
    Applies the transformation matrix to the mask.

    Parameters
    ----------
    fixed_image_path : str
        Path to the fixed image.
    moving_mask_path : str
        Path to the moving mask.
    output_mask_path : str
        Path to the output mask.
    transformation_file : str, optional
        Path to the transformation matrix. The default is None.
    remove_transformation : bool, optional
        If True, the transformation matrix and all the transformation files are removed. The default is True.

    Raises
    ------
    TypeError
        If the transformation file is not provided and the default path does not exist.
    """
    if transformation_file is None:
        if not os.path.exists(os.path.join(os.path.dirname(output_mask_path), 'transfomation_matrix.txt')):
            raise TypeError("Text file with transformation path does not exist.")       
        else:
            transformation_file = os.path.join(os.path.dirname(output_mask_path), 'transfomation_matrix.txt')

    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_mask_path)
    
    # Load the transformation matrix
    if transformation_file.endswith("txt"):
        with open(transformation_file, 'r') as f:
            transformations = [line.strip() for line in f]
    else:
        transformations = [transformation_file]
      
    # Apply the transformation to the mask
    warped_mask = ants.apply_transforms(fixed = fixed_image, 
                                        moving = moving_image, 
                                        transformlist = transformations, 
                                        interpolator='genericLabel')
    
    # Save the transformed mask
    warped_mask.to_file(output_mask_path)
    
    if remove_transformation:
        for transformation_path in transformations:
            os.remove(transformation_path)
        os.remove(transformation_file)
    
def convert_to_orientation(nifti, reference_ornt = ("R", "A", "S")):
    """
    Converts nifti orientation to that of reference.

    Parameters
    ----------
    nifti : nibabel.nifti1.Nifti1Image
        Nifti image to be converted.
    reference_ornt : tuple, optional
        Orientation to convert to. The default is ("R", "A", "S").

    Returns
    -------
    reoriented_nifti : nibabel.nifti1.Nifti1Image
        Reoriented nifti image.
    """
    # Get orientation from affine matrices
    nifti_ornt = nib.aff2axcodes(nifti.affine)

    if nifti_ornt == reference_ornt:
        print("Nifti orientation already in {}".format(reference_ornt))
        return nifti
    else:
        print("Converting orientation from {} to {}".format(nifti_ornt, reference_ornt))
        data = nifti.get_fdata()
        affine = nifti.affine
        # Get orientation transform from axcodes
        ornt = nib.orientations.axcodes2ornt(reference_ornt)
        # Apply orientation transform to array
        data = nib.orientations.apply_orientation(data, ornt)
        # Update affine matrix
        for idx in range(len(nifti_ornt)):
            if nifti_ornt[idx] != reference_ornt[idx]:
                affine[idx, 3] = affine[idx, 3] + affine[idx, idx] * (data.shape[idx] - 1)
                affine[idx, idx] = -affine[idx, idx]

        reoriented_nifti = nib.Nifti1Image(data, affine=affine)

        return reoriented_nifti

def mat_to_affine(mat_path):
    """
    Reads the affine transformation from a .mat file and returns it as a numpy array.

    Parameters
    ----------
    mat_path : str
        Path to .mat file.
        
    Returns
    -------
    affine : numpy.ndarray
        Affine matrix.
    """
    mat = sio.loadmat(mat_path)
    rotation_scaling = np.array(mat["AffineTransform_float_3_3"]).reshape(3, 4)
    translation = np.array(mat["fixed"]).reshape(3, 1)
    affine = np.eye(4)
    affine[:3, :4] = rotation_scaling
    affine[:3, 3] = translation.ravel()
    return affine

def resample_nifti(nifti, target_spacing):
    """
    Resamples the nifti image to the target spacing.

    Parameters
    ----------
    nifti : nibabel.nifti1.Nifti1Image
        Nifti image to be resampled.
    target_spacing : float or array_like
        Target spacing of the resampled image.  
        If float, isotropic spacing is used.
        If array_like, spacing is used for each axis.
    
    Returns
    -------
    new_img : nibabel.nifti1.Nifti1Image
        Resampled nifti image.
    """
    data = nifti.get_fdata()
    header = nifti.header
    affine = nifti.affine

    # Original spacing (from the affine matrix)
    original_spacing = nifti.header.get_zooms()
    # If target_spacing is a single value, make it isotropic
    if isinstance(target_spacing, float):
        target_spacing = np.array([target_spacing, target_spacing, target_spacing])

    # Calculate zoom factors
    zoom_factors = original_spacing / target_spacing

    # Resample the image using 3rd order spline interpolation
    resampled_data = zoom(data, zoom_factors, order=5)

    # Update the header and affine matrix
    new_affine = np.copy(affine)
    np.fill_diagonal(new_affine, np.append(target_spacing, [1]))
    for idx in range(3):
        new_affine[idx, idx] *= np.sign(affine[idx, idx])
    new_img = nib.Nifti1Image(resampled_data, new_affine, header)

    return new_img

def resample_atlas_to_median_spacing(atlas_path, db):
    """
    Resamples the atlas to the median spacing of the dataset. Saves the resampled 
    atlas in the same directory as the original atlas, with the "median_spacing"
    suffix.

    Parameters
    ----------
    atlas_path : str
        Path to the atlas.
    db : str
        Path to the database.

    """
    # Compute median spacing of the dataset
    median_spacing = compute_median_spacing(db)

    # Load atlas
    atlas = nib.load(atlas_path)
    # Resample atlas to median spacing
    resampled_atlas = resample_nifti(atlas, median_spacing)

    # Save resampled atlas
    nib.save(resampled_atlas, os.path.join(db, os.path.dirname(atlas_path), "mni_atlas_median_spacing.nii.gz"))

def compute_median_spacing(db):
    """
    Computes the median spacing of the dataset.

    Parameters
    ----------
    db : str
        Path to the database.

    Returns
    -------
    median_spacing : numpy.ndarray
        Median spacing of the dataset.
    """
    # Read cases from a database
    case_ids = sorted([case_id for case_id in os.listdir(db)])
    # Initialize array for median spacing
    median_spacing = np.ndarray([len(case_ids), 3])
    # Store spacing for each case from the CTA
    for idx, case_id in enumerate(case_ids):
        cta = nib.load(os.path.join(db, case_id, f"{case_id}_cta.nii.gz"))
        spacing = np.array(cta.header.get_zooms())
        median_spacing[idx, :] = spacing

    # Return median spacing
    return np.median(median_spacing, axis=0)