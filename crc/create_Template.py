import os
import cv2


def yolo_to_bbox(label_line, image_width, image_height):
    """
    YOLO format:
    class_id x_center y_center width height
    The values are normalized between 0 and 1.
    This function converts them to pixel coordinates.
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

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width - 1, x2)
    y2 = min(image_height - 1, y2)

    return class_id, x1, y1, x2, y2


def create_templates(images_dir, labels_dir, templates_dir):
    os.makedirs(templates_dir, exist_ok=True)

    image_files = [
        file for file in os.listdir(images_dir)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".ppm"))
    ]

    print(f"Found {len(image_files)} images.")

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image_name_without_extension = os.path.splitext(image_file)[0]
        label_path = os.path.join(labels_dir, image_name_without_extension + ".txt")

        if not os.path.exists(label_path):
            print(f"No label found for {image_file}")
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read image: {image_file}")
            continue

        image_height, image_width = image.shape[:2]

        with open(label_path, "r") as label_file:
            lines = label_file.readlines()

        for index, line in enumerate(lines):
            if line.strip() == "":
                continue

            class_id, x1, y1, x2, y2 = yolo_to_bbox(
                line,
                image_width,
                image_height
            )

            cropped_sign = image[y1:y2, x1:x2]

            if cropped_sign.size == 0:
                print(f"Empty crop in {image_file}")
                continue

            class_folder = os.path.join(templates_dir, f"class_{class_id}")
            os.makedirs(class_folder, exist_ok=True)

            template_name = f"{image_name_without_extension}_{index}.jpg"
            template_path = os.path.join(class_folder, template_name)

            cv2.imwrite(template_path, cropped_sign)

    print("Done. Templates were created successfully.")


if __name__ == "__main__":
    create_templates(
        images_dir="archive (1)\GTSDB_Train_and_Test\Train\images",
        labels_dir="archive (1)\GTSDB_Train_and_Test\Train\labels",
        templates_dir="templates"
    )