import os
import cv2
from orbDetectors import detect_with_orb, draw_detection

images_dir = r"archive (1)\GTSDB_Train_and_Test\Train\images"
labels_dir = r"archive (1)\GTSDB_Train_and_Test\Train\labels"
templates_dir = "templates"
output_dir = r"results\detections_orb"

os.makedirs(output_dir, exist_ok=True)


def get_image_files(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".ppm"))
    ]


def get_template_files(folder):
    template_paths = []

    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)

        if not os.path.isdir(class_folder):
            continue

        files = [
            f for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".ppm"))
        ]

        # use only first 3 templates per class to make it faster
        for file in files[:3]:
            template_paths.append(os.path.join(class_folder, file))

    return template_paths


image_files = get_image_files(images_dir)
template_files = get_template_files(templates_dir)

image_files = image_files[:20]

total_images = 0
images_with_labels = 0
detected_images = 0

print("Images:", len(image_files))
print("Templates:", len(template_files))

for image_file in image_files:
    total_images += 1

    scene_path = os.path.join(images_dir, image_file)
    scene = cv2.imread(scene_path)

    if scene is None:
        continue

    label_path = os.path.join(
        labels_dir,
        os.path.splitext(image_file)[0] + ".txt"
    )

    if os.path.exists(label_path):
        images_with_labels += 1

    found = False
    output = scene.copy()

    print("\nProcessing:", image_file)

    for template_path in template_files:
        template = cv2.imread(template_path)

        if template is None:
            continue

        detected, corners, matches = detect_with_orb(
            template_image=template,
            scene_image=scene,
            min_matches=4
        )

        if detected:
            print("Detected:", os.path.basename(template_path), "Matches:", matches)

            output = draw_detection(output, corners, matches)
            found = True
            break

    if found:
        detected_images += 1
    else:
        cv2.putText(
            output,
            "No detection",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imwrite(os.path.join(output_dir, image_file), output)

print("\nORB Results")
print("Total images tested:", total_images)
print("Images with traffic signs:", images_with_labels)
print("Detected images:", detected_images)

if images_with_labels > 0:
    recall = detected_images / images_with_labels
    print("Detection rate / Recall:", round(recall * 100, 2), "%")

print("\nDone. Check:", output_dir)