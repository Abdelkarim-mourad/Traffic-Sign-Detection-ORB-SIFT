import os
import cv2
from SiftDetector import detect_with_sift, draw_detection


template_path = r"templates\class_1\00225_0.jpg"
scene_path = r"archive (1)\GTSDB_Train_and_Test\Train\images\00225.jpg"
template = cv2.imread(template_path)
scene = cv2.imread(scene_path)

if template is None:
    raise FileNotFoundError(f"Could not read template: {template_path}")

if scene is None:
    raise FileNotFoundError(f"Could not read scene image: {scene_path}")

detected, corners, matches_count = detect_with_sift(
    template_image=template,
    scene_image=scene,
    min_matches=6
)

print("Detected:", detected)
print("Good matches:", matches_count)

if detected:
    output = draw_detection(scene, corners, matches_count)
else:
    output = scene.copy()
    cv2.putText(
        output,
        "No detection",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

os.makedirs(r"..\Resultats\detections_sift", exist_ok=True)

output_path = r"..\Resultats\detections_sift\sift_result.jpg"
cv2.imwrite(output_path, output)

print("Saved result to:", output_path)