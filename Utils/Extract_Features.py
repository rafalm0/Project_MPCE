import glob
import os
import numpy as np
import face_recognition
import cv2
from imutils.face_utils import FaceAligner
import dlib
import pandas as pd
import pickle


def extract_face_features(images_train_path: str,
                          trained_prediciton_path: str = "/opt/project/Utils/pretrained_models/shape_predictor_68_face_landmarks.dat",
                          normalize: bool = True, print_key: bool = True, save_pickle: bool = True,
                          save_csv: bool = True,
                          detection_method: str = "cnn",
                          plot_faces: bool = False) -> pd.DataFrame:
    predictor = dlib.shape_predictor(trained_prediciton_path)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    images_path = glob.glob(f"{images_train_path}/*")
    data = {"imagePath": [], "face_locations": [], "encoding": []}
    for (i, imagePath) in enumerate(images_path):
        if print_key:
            print(f"[INFO] processing {imagePath} , {i + 1}/{len(images_path)}")
            print(imagePath)

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        boxes = face_recognition.face_locations(rgb, model=detection_method)
        if len(boxes) == 0:
            continue

        if normalize:
            encodings = []

            for i, box in enumerate(boxes):
                top, right, bottom, left = box
                rec = dlib.rectangle(top=top, right=right, bottom=bottom, left=left)
                faceAligned = fa.align(image, gray, rec)
                faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB)
                # faceAligned = imutils.resize(faceAligned, width= right - left, height= bottom - top)
                # if plot_faces:
                #   cv2_imshow(faceAligned)
                # cv2_imshow(a)

                encodings.extend(face_recognition.face_encodings(faceAligned, [(0, 256, 256, 0)]))
        else:
            encodings = face_recognition.face_encodings(rgb, boxes)

        for (box, enc) in zip(boxes, encodings):
            data["imagePath"].append(imagePath)
            data["face_locations"].append(box)
            data["encoding"].append(enc)

    data = pd.DataFrame(data)
    data.index += 1
    exit_path ="/".join(images_train_path.split("/")[:-1])
    if save_csv:
        data.to_csv(f"{exit_path}/image_encondings.csv")
    if save_pickle:
        save_pickle_at(data, exit_path)

    return data


def save_pickle_at(data, path):
    f = open(f"{path}/image_encondings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()


if __name__ == '__main__':
    # print(os.path.abspath(__file__))
    path = "/opt/project/dataset/train"
    # print(os.listdir(path))
    df = extract_face_features(path)
    print(df)
