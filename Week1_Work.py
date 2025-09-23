import cv2
import numpy as np
import os

image_path = input("Enter image path (e.g., D:/img/test.jpg): ").strip()
if not image_path or not os.path.isfile(image_path):
    print("File not found."); raise SystemExit

try:
    K = int(input("Enter K (number of colors) [default 4]: ") or 4)
except:
    K = 4

try:
    MIN_AREA = int(input("Ignore blobs smaller than area [default 80]: ") or 80)
except:
    MIN_AREA = 80

img = cv2.imread(image_path)
if img is None:
    print("Failed to read image."); raise SystemExit

h, w = img.shape[:2]
data = np.float32(img.reshape(-1, 3))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(data, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
labels = labels.reshape(h, w)
centers = np.uint8(centers)

stem = os.path.splitext(os.path.basename(image_path))[0]
out_dir = os.path.join("output_beginner_en", stem)
os.makedirs(out_dir, exist_ok=True)

masks = []
only_images = []
for i in range(K):
    mask = (labels == i).astype("uint8") * 255
    masks.append(mask)
    only = np.zeros_like(img)
    only[labels == i] = img[labels == i]
    only_images.append(only)
    cv2.imwrite(os.path.join(out_dir, f"cluster_{i:02d}_mask.png"), mask)
    cv2.imwrite(os.path.join(out_dir, f"cluster_{i:02d}_only.png"), only)

annotated = img.copy()

for i, mask in enumerate(masks):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    print(f"\nCluster {i}:")
    idx = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        idx += 1
        print(f"  Object {idx}: center=({cx}, {cy}), area={int(area)}")
        cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(annotated, f"C{i}-O{idx}", (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"C{i}-O{idx}", (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imwrite(os.path.join(out_dir, "annotated.png"), annotated)
print("\nSaved to:", out_dir)
print("Files: cluster_xx_mask.png, cluster_xx_only.png, annotated.png")

show = input("\nPreview windows? (y/N): ").strip().lower()
if show == "y":
    cv2.imshow("Original", img)
    cv2.imshow("Annotated", annotated)
    for i in range(K):
        cv2.imshow(f"mask_{i}", masks[i])
        cv2.imshow(f"only_{i}", only_images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
