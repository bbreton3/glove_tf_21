import os
import glob

def test_save_model_callback(save_model_folder_path):

    checkpoint_file = os.path.join(save_model_folder_path, "checkpoint")
    assert os.path.isfile(checkpoint_file)
    assert os.path.isfile(os.path.join(save_model_folder_path, "ckpt.last_epoch_number.txt"))
    assert os.path.isfile(os.path.join(save_model_folder_path, "ckpt.index"))
    assert len(glob.glob(os.path.join(save_model_folder_path, "ckpt.data*"))) > 0

    os.remove(checkpoint_file)
    for file_path in glob.glob(os.path.join(save_model_folder_path, "ckpt.*")):
        os.remove(file_path)