import cv2
import numpy as np


# ----------------------------------------------------------------------
# ORB configuration  (values chosen from an actual parameter sweep on
# GTSDB, not from theory -- see notes)
# ----------------------------------------------------------------------
# Findings from the sweep:
#  - Traffic signs are SMALL and LOW-TEXTURE. Keep edgeThreshold SMALL
#    (5): a large edgeThreshold discards the border keypoints and leaves
#    almost nothing on a 50px template. (Setting edgeThreshold==patchSize,
#    the "textbook" rule, killed ALL detections here -> DON'T.)
#  - A slightly smaller patchSize (21) fits tiny templates better than 31.
#  - A lower fastThreshold (12) recovers more corners on low-contrast signs.
#  - DO NOT upscale the template: it shifts the scale relationship and
#    actually reduced true matches in testing.
#  - Keep the ratio test STRICT (0.75). Loosening to 0.8 produced 100+
#    "good" matches that were almost all FALSE (boxes landed on background,
#    IoU 0.0 with the real sign). More matches != better.
#  - The REAL detection gate is post-RANSAC INLIERS, not raw good matches.
# ----------------------------------------------------------------------

TEMPLATE_UPSCALE = 1.0     # do NOT upscale (hurt matching in testing)
ORB_NFEATURES = 3000
ORB_SCALE_FACTOR = 1.2
ORB_NLEVELS = 8
ORB_PATCH_SIZE = 21
ORB_EDGE_THRESHOLD = 5     # small on purpose: keep keypoints on tiny signs
ORB_FAST_THRESHOLD = 12

RATIO = 0.75               # Lowe ratio test (strict = high precision)
MIN_GOOD_MATCHES = 12      # need enough for a stable homography
MIN_INLIERS = 10           # post-RANSAC inliers (the REAL gate)
MIN_INLIER_RATIO = 0.30


def build_orb():
    return cv2.ORB_create(
        nfeatures=ORB_NFEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_NLEVELS,
        edgeThreshold=ORB_EDGE_THRESHOLD,
        patchSize=ORB_PATCH_SIZE,
        fastThreshold=ORB_FAST_THRESHOLD,
    )


def compute_scene_features(scene_image, orb=None):
    """Compute ORB features for a scene ONCE and reuse across templates.

    The original runOrbMultiple.py recomputed 5000 scene keypoints for
    every single template (43 templates x 20 images = 860 times). That is
    extremely wasteful. Compute once, pass in.
    """
    if orb is None:
        orb = build_orb()
    scene_gray = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(scene_gray, None)
    return kp, des


def _template_features(template_image, orb):
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    if TEMPLATE_UPSCALE != 1.0:
        template_gray = cv2.resize(
            template_gray, None,
            fx=TEMPLATE_UPSCALE, fy=TEMPLATE_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )
    kp, des = orb.detectAndCompute(template_gray, None)
    # scale keypoint coords back to the ORIGINAL template size so the
    # homography maps the correct template rectangle.
    for k in kp:
        k.pt = (k.pt[0] / TEMPLATE_UPSCALE, k.pt[1] / TEMPLATE_UPSCALE)
    return kp, des, template_gray


def detect_with_orb(template_image, scene_image,
                    scene_kp=None, scene_des=None,
                    min_matches=MIN_GOOD_MATCHES,
                    verbose=False):
    """Detect a single template inside a scene.

    Pass scene_kp / scene_des if you already computed them once for this
    scene (recommended in the multi loop).

    Returns (detected, scene_corners, good_match_count).
    """
    orb = build_orb()

    kp1, des1, template_gray = _template_features(template_image, orb)

    if scene_kp is None or scene_des is None:
        scene_kp, scene_des = compute_scene_features(scene_image, orb)
    kp2, des2 = scene_kp, scene_des

    if verbose:
        print("Template keypoints:", 0 if kp1 is None else len(kp1))
        print("Scene keypoints:", 0 if kp2 is None else len(kp2))

    if des1 is None or des2 is None or len(kp1) < 2:
        return False, None, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO * n.distance:
            good_matches.append(m)

    if verbose:
        print("Good matches:", len(good_matches))

    if len(good_matches) < min_matches:
        return False, None, len(good_matches)

    src_points = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_points = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return False, None, len(good_matches)

    inliers = int(mask.sum())
    inlier_ratio = inliers / len(good_matches)

    if verbose:
        print("Inliers:", inliers, "ratio:", round(inlier_ratio, 3))

    # The REAL detection gate is inliers (not raw good matches). With only
    # 4 matches the homography is exactly determined and RANSAC cannot
    # reject anything -> that is exactly why min_matches=4 produced wrong
    # boxes in the original code.
    if inliers < MIN_INLIERS:
        return False, None, len(good_matches)
    if inlier_ratio < MIN_INLIER_RATIO:
        return False, None, len(good_matches)

    h, w = template_gray.shape
    h = int(h / TEMPLATE_UPSCALE)
    w = int(w / TEMPLATE_UPSCALE)

    template_corners = np.float32(
        [[0, 0], [w, 0], [w, h], [0, h]]
    ).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(template_corners, H)

    if not is_valid_homography_box(scene_corners, scene_image, verbose):
        return False, None, len(good_matches)

    return True, scene_corners, len(good_matches)


def is_valid_homography_box(corners, scene_image, verbose=False):
    image_height, image_width = scene_image.shape[:2]
    points = corners.reshape(4, 2)

    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    box_width = x_max - x_min
    box_height = y_max - y_min

    aspect_ratio = box_width / float(box_height + 1e-6)
    area = cv2.contourArea(points.astype(np.float32))

    if verbose:
        print("Box:", x_min, y_min, x_max, y_max,
              "AR:", round(aspect_ratio, 2), "area:", round(area, 1))

    # A valid homography for a planar sign must stay convex. A collapsed /
    # self-intersecting quad is the classic "box drawn wrong" symptom.
    if not _is_convex(points):
        return False

    if x_min < 0 or y_min < 0:
        return False
    if x_max > image_width or y_max > image_height:
        return False
    if box_width < 8 or box_height < 8:
        return False
    if box_width > image_width * 0.35 or box_height > image_height * 0.35:
        return False
    if aspect_ratio > 2.0 or aspect_ratio < 0.5:   # signs are ~square
        return False
    if area < 200:
        return False
    return True


def _is_convex(points):
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    return cv2.isContourConvex(pts)


def draw_detection(scene_image, corners, matches_count, label="sign"):
    output = scene_image.copy()
    corners_int = np.int32(corners)
    cv2.polylines(output, [corners_int], True, (0, 255, 0), 2)

    points = corners.reshape(4, 2)
    x_min = int(np.min(points[:, 0]))
    y_min = int(np.min(points[:, 1]))

    cv2.putText(
        output,
        f"{label} | ORB matches: {matches_count}",
        (x_min, max(y_min - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
    )
    return output