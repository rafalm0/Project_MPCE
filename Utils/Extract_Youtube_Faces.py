import tarfile
import os
import glob
import zipfile
import shutil


def unzip_tar_gz(file_path: str, exit_path: str = ""):
    if not os.path.exists(exit_path):
        os.mkdir(exit_path)

    tar = tarfile.open(file_path, "r:gz")
    tar.extractall(exit_path)
    tar.close()


def extract_files(imgs_path: str, one_example: bool = True):
    train_path = "dataset/train"
    test_path = "dataset/test"

    if not os.path.exists("dataset"):
        os.mkdir("dataset")
    if not os.path.exists("dataset/train"):
        os.mkdir("dataset/train")
    if not os.path.exists("dataset/test"):
        os.mkdir("dataset/test")

    for person in os.listdir(f"{imgs_path}"):
        if not (person.endswith(".txt") and person.endswith(".zip")):
            person_path = f"{imgs_path}/{person}"
            for j in glob.glob(f"{person_path}/*"):
                files = os.listdir(f"{j}")

                position = len(files) - 1 if one_example else int(len(files) * 0.8) + 1

                for z in files[:position]:
                    file_path = f"{j}/{z}"
                    shutil.move(file_path, train_path)
                    os.rename(f"{train_path}/{z}", f"{train_path}/{person}_{z}")

                for z in files[position:]:
                    file_path = f"{j}/{z}"
                    shutil.move(file_path, test_path)
                    os.rename(f"{test_path}/{z}", f"{test_path}/{person}_{z}")

        # if os.path.getsize("dataset/train") > 6000000000:
        #     break


if __name__ == '__main__':
    file_path = "E:\Download\YoutubeFacesDataset\YouTubeFaces.tar.gz"
    exit_path = "dataset"
    unzip_tar_gz(file_path, exit_path = exit_path)
    extracted_path = f"{exit_path}/YouTubeFaces/frame_images_DB"
    extract_files(extracted_path)
