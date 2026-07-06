import os
import cv2
from orbDetectors import detect_with_orb, draw_detection, compute_scene_features

ARCHIVE = os.path.join("archive (1)", "GTSDB_Train_and_Test", "Train")

template_path = os.path.join("templates", "class_1", "00012_0.jpg")
scene_path = os.path.join(ARCHIVE, "images", "00012.jpg")

template = cv2.imread(template_path)
scene = cv2.imread(scene_path)

if template is None:
    raise FileNotFoundError(f"Could not read template: {template_path}")
if scene is None:
    raise FileNotFoundError(f"Could not read scene image: {scene_path}")

# Compute scene features once (good habit, and required by the fast API).
scene_kp, scene_des = compute_scene_features(scene)

detected, corners, matches_count = detect_with_orb(
    template_image=template,
    scene_image=scene,
    scene_kp=scene_kp,
    scene_des=scene_des,
    verbose=True,
)

print("Detected:", detected)
print("Good matches:", matches_count)

if detected:
    output = draw_detection(scene, corners, matches_count, label="class_1")
else:
    output = scene.copy()
    cv2.putText(output, "No detection", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

os.makedirs("Resultats/detections_orb", exist_ok=True)
output_path = "Resultats/detections_orb/test_result.jpg"
cv2.imwrite(output_path, output)
print("Saved result to:", output_path)