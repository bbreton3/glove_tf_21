import glob
import os
import shutil


def test_lr_tensorboard_callback(tensorboard_folder_path):
    train_tensorboard_folder = os.path.join(tensorboard_folder_path, "train")
    assert os.path.isdir(train_tensorboard_folder)
    assert len(glob.glob(os.path.join(train_tensorboard_folder, "events.out.tfevents.*"))) > 1

    assert "epoch_lr" in str(
        open(glob.glob(os.path.join(train_tensorboard_folder, "events.out.tfevents.*.v2"))[0], "rb").read())

    shutil.rmtree(train_tensorboard_folder)
