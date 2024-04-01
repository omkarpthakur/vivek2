import cv2
import os
import numpy as np

def stitch_images(image_paths, output_path):
    if len(image_paths) == 1:
        return cv2.imread(image_paths[0])

    if len(image_paths) <= 10:
        return stitch_batch(image_paths)

    # If more than 10 images, stitch in batches of 10
    stitched_batches = []
    for i in range(0, len(image_paths), 10):
        batch = image_paths[i:i+10]
        stitched_batches.append(stitch_batch(batch))

    final_stitched_image = stitch_images(stitched_batches, None)
    if output_path:
        cv2.imwrite(output_path, final_stitched_image)
    return final_stitched_image

def stitch_batch(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"Error: Unable to read image from {path}")
            continue
        images.append(image)

    if not images:
        print("Error: No valid images found in the batch.")
        return None

    # Convert images to grayscale
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Find keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    for gray_image in gray_images:
        kp, desc = sift.detectAndCompute(gray_image, None)
        keypoints.append(kp)
        descriptors.append(desc)

    # Match keypoints
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    all_matches = []
    for i in range(len(descriptors) - 1):
        matches = matcher.knnMatch(descriptors[i], descriptors[i+1], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        all_matches.append(good_matches)

    # Extract matched keypoints
    src_points = []
    dst_points = []
    for matches, kp1, kp2 in zip(all_matches, keypoints[:-1], keypoints[1:]):
        src_pts = np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp.pt for kp in kp2]).reshape(-1, 1, 2)
        src_points.append(src_pts)
        dst_points.append(dst_pts)

    # Find homography matrices
    homographies = []
    for src, dst in zip(src_points, dst_points):
        homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        homographies.append(homography)

    # Warp images and stitch
    stitched_images = [images[0]]
    for i in range(1, len(images)):
        height, width = images[i].shape[:2]
        warped_image = cv2.warpPerspective(images[i], homographies[i-1], (width, height))
        stitched_images.append(warped_image)

    stitched_image = stitched_images[0]
    for i in range(1, len(stitched_images)):
        stitched_image = cv2.addWeighted(stitched_image, 0.5, stitched_images[i], 0.5, 0)

    return stitched_image

# Example usage:
image_dir = r'D:\vivekcv\cg_project\Scripts\image_directory'  # Make sure to use a raw string
output_path = r'D:\vivekcv\cg_project\Scripts\output\stitched_image.jpg'  # Output path for stitched image
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
stitched_image = stitch_images(image_paths, output_path)
if stitched_image is not None:
    print(f"Stitched image saved at: {output_path}")
