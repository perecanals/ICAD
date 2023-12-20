import os, shutil

from icad.preprocessing.preprocessing import preprocess_case
from arterial.io.load_and_save_operations import load_json, save_json 
from time import time

base_dir = os.environ["icad_base_path"]

if os.path.exists(os.path.join(base_dir, "times_preprocessing.json")):
    times_log = load_json(os.path.join(base_dir, "times_preprocessing.json"))
else:
    times_log = {}

if os.path.exists(os.path.join(base_dir, "errors_preprocessing.json")):
    errors_log = load_json(os.path.join(base_dir, "errors_preprocessing.json"))
else:
    errors_log = {}

# Define relevant paths
atlas_path = os.path.join(base_dir, "atlas/mni_atlas_median_spacing.nii.gz")
original_db_path = os.path.join(base_dir, "database/occlusion")

for case_id in sorted([case_id for case_id in os.listdir(original_db_path)]):
    if case_id not in times_log and case_id not in errors_log:
        try:
            if not os.path.exists(os.path.join(original_db_path, case_id, f"{case_id}_ncct.nii.gz")):
                shutil.copy(os.path.join(original_db_path, case_id, f"{case_id}_ncct_reg.nii.gz"), 
                            os.path.join(original_db_path, case_id, f"{case_id}_ncct.nii.gz"))
            if not os.path.exists(os.path.join(original_db_path, case_id, f"{case_id}_extracranial_vessels_segmentation.nii.gz")):
                shutil.copy(os.path.join(original_db_path, case_id, f"{case_id}_segmentation.nii.gz"), 
                            os.path.join(original_db_path, case_id, f"{case_id}_extracranial_vessels_segmentation.nii.gz"))
            start = time()
            # Perform preprocessing
            preprocess_case(case_id=case_id,
                            atlas_path=atlas_path,
                            cta_path=os.path.join(original_db_path, case_id, f"{case_id}_cta.nii.gz"),
                            ncct_path=os.path.join(original_db_path, case_id, f"{case_id}_ncct.nii.gz"),
                            vessel_segmentation_path=os.path.join(original_db_path, case_id, f"{case_id}_extracranial_vessels_segmentation.nii.gz"),
                            thrombus_segmentation_path=os.path.join(original_db_path, case_id, f"{case_id}_thrombus_segmentation.nii.gz"))

            times_log[case_id] = time() - start
            save_json(times_log, os.path.join(base_dir, "times_preprocessing.json"))
        except Exception as e:
            errors_log[case_id] = str(e)
            print(f"Error preprocessing case {case_id}: {e}")
            save_json(errors_log, os.path.join(base_dir, "errors_preprocessing.json"))
            continue