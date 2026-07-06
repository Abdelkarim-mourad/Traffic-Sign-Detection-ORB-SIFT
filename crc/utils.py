import os
import cv2


def yolo_to_bbox(label_line, image_width, image_height):
    """
    Converts YOLO label format to pixel bounding box.
    YOLO format:
    class_id x_center y_center width height
    """
    parts = label_line.strip().split()

    class_id = int(parts[0])
    x_center = float(parts[1]) * image_width
    y_center = float(parts[2]) * image_height
    box_width = float(parts[3]) * image_width
    box_height = float(parts[4]) * image_height

    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    # Keep coordinates inside image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width - 1, x2)
    y2 = min(image_height - 1, y2)

    return class_id, x1, y1, x2, y2


def get_matching_label_path(image_path, labels_dir):
    """
    Example:
    images/00001.jpg -> labels/00001.txt
    """
    image_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(image_name)[0]
    label_path = os.path.join(labels_dir, name_without_ext + ".txt")

    return label_path


def draw_bbox(image, bbox, label_text="sign"):
    x1, y1, x2, y2 = bbox

    output = image.copy()

    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        output,
        label_text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    return output