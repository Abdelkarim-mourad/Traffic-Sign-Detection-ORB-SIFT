import cv2
import numpy as np


def detect_with_sift(template_image, scene_image, min_matches=6):
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(scene_gray, None)

    print("Template keypoints:", 0 if kp1 is None else len(kp1))
    print("Scene keypoints:", 0 if kp2 is None else len(kp2))

    if des1 is None or des2 is None:
        print("Reason: No descriptors")
        return False, None, 0

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print("Good matches:", len(good_matches))

    if len(good_matches) < min_matches:
        print("Reason: Not enough good matches")
        return False, None, len(good_matches)

    src_points = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    dst_points = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    if H is None:
        print("Reason: Homography failed")
        return False, None, len(good_matches)

    h, w = template_gray.shape

    corners = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ]).reshape(-1, 1, 2)

    scene_corners = cv2.perspectiveTransform(corners, H)

    return True, scene_corners, len(good_matches)


def draw_detection(scene_image, corners, matches_count):
    output = scene_image.copy()

    corners = np.int32(corners)
    cv2.polylines(output, [corners], True, (255, 0, 0), 2)

    points = corners.reshape(4, 2)

    x_min = int(np.min(points[:, 0]))
    y_min = int(np.min(points[:, 1]))

    cv2.putText(
        output,
        f"SIFT matches: {matches_count}",
        (x_min, max(y_min - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    return output