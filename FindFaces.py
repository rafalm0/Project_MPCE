import face_recognition
import cv2


def find_faces(images_path: [], detection_method: str = "cnn", generate_encoding: bool = True) -> list:
    faces = []
    for (i, image_path) in enumerate(images_path):
        print(f"[INFO] processing {image_path} , {i + 1}/{len(images_path)}")

        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model=detection_method)

        if generate_encoding:
            encodings = face_recognition.face_encodings(rgb, boxes)
            d = [{"imagePath": image_path, "loc": box,
                  "center": rectangle_center_position((box[0], box[3]), (box[2], box[1])), "encoding": enc}
                 for (box, enc) in zip(boxes, encodings)]
        else:
            d = [{"imagePath": image_path, "loc": box} for box in boxes]

        faces.extend(d)

    return faces


def rectangle_center_position(start_point: tuple, end_point: tuple) -> tuple:
    return int((end_point[0] - start_point[0]) / 2) + start_point[0], int((end_point[1] - start_point[1]) / 2) + \
           start_point[1]


# if __name__ == '__main__':
#     info = find_faces(["00000002.jpg"])
#     image = cv2.imread("00000002.jpg")
#     print(info[0]["loc"])
#     print(len(info))
#     (top, right, bottom, left) = info[0]["loc"]
#
#     start_point = top, left
#     end_point = bottom, right
#
#     center = info[0]["center"]
#     thickness = 2
#     color = (255, 0, 0)
#     cv2.rectangle(image, pt1=(start_point[1], start_point[0]), pt2=(end_point[1], end_point[0]), color=color,
#                   thickness=thickness)
#
#     print(start_point, end_point)
#     print(center)
#     image[start_point] = (0, 255, 0)
#     image[end_point] = (0, 255, 0)
#     image[center] = (255, 0, 0)
#     cv2.imwrite("ts.jpg", image)
