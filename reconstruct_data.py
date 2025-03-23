import os
import shutil

def restructure_mura_by_bone(base_path, output_path):
    # Define train/valid directories
    for split in ["train", "valid"]:
        input_dir = os.path.join(base_path, split)
        output_dir = os.path.join(output_path, split)

        # Iterate through modality folders (e.g., XR_WRIST, XR_ELBOW, etc.)
        for modality in os.listdir(input_dir):
            modality_path = os.path.join(input_dir, modality)
            if os.path.isdir(modality_path):
                # Create bone-specific folders within train/valid
                bone_type = modality.split("_")[1].lower()  # Extract "wrist", "elbow", etc.
                bone_output_dir = os.path.join(output_dir, bone_type)
                os.makedirs(bone_output_dir, exist_ok=True)

                # Create positive and negative folders within bone-specific folders
                pos_dir = os.path.join(bone_output_dir, "positive")
                neg_dir = os.path.join(bone_output_dir, "negative")
                os.makedirs(pos_dir, exist_ok=True)
                os.makedirs(neg_dir, exist_ok=True)

                for patient in os.listdir(modality_path):
                    patient_path = os.path.join(modality_path, patient)
                    if os.path.isdir(patient_path):
                        for study in os.listdir(patient_path):
                            study_path = os.path.join(patient_path, study)
                            if os.path.isdir(study_path):
                                # Determine if study is positive or negative
                                if "positive" in study:
                                    target_folder = pos_dir
                                else:
                                    target_folder = neg_dir

                                # Move all images into corresponding class folder
                                for image in os.listdir(study_path):
                                    image_path = os.path.join(study_path, image)
                                    new_image_path = os.path.join(target_folder, f"{patient}_{study}_{image}")

                                    if not os.path.exists(new_image_path):  # Avoid overwriting
                                        shutil.copy2(image_path, new_image_path)

    print(f"MURA dataset successfully restructured by bone type in: {output_path}")
