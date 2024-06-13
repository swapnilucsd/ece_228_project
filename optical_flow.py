import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    return cv2.imread(path)


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def draw_optical_flow_vectors(image, points1, points2):
    for pt1, pt2 in zip(points1, points2):
        pt1 = tuple(np.round(pt1).astype(int))
        pt2 = tuple(np.round(pt2).astype(int))
        image = cv2.arrowedLine(image, pt1, pt2, color=(255, 0, 0), thickness=1)
    return image


def feature_matching_and_flow(frame1, frame2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(frame1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(frame2, None)
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    return points1, points2


def compute_cv_for_tile(flow_vectors):
    if len(flow_vectors) == 0:
        return None
    magnitudes = np.linalg.norm(flow_vectors, axis=1)
    mean_flow = np.mean(magnitudes)
    std_flow = np.std(magnitudes)
    cv = std_flow / mean_flow
    return cv


def calculate_dense_optical_flow(frame1, frame2):
    gray1 = convert_to_grayscale(frame1)
    gray2 = convert_to_grayscale(frame2)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def analyze_optical_flow(flow, tile_width, tile_height):
    h, w = flow.shape[:2]
    cv_values = np.zeros((h // tile_height, w // tile_width))
    for i in range(0, h, tile_height):
        for j in range(0, w, tile_width):
            tile_flows = flow[i : i + tile_height, j : j + tile_width].reshape(-1, 2)
            cv = compute_cv_for_tile(tile_flows)
            cv_values[i // tile_height, j // tile_width] = cv if cv is not None else 0
    return cv_values > 0.1


def outputimage(flat_areas, frame2, tile_width, tile_height):
    output_image = frame2.copy()
    for i in range(flat_areas.shape[0]):
        for j in range(flat_areas.shape[1]):
            if flat_areas[i, j]:
                top_left = (j * tile_width, i * tile_height)
                bottom_right = ((j + 1) * tile_width, (i + 1) * tile_height)
                cv2.rectangle(output_image, top_left, bottom_right, (0, 255, 0), 2)

    return output_image


def display_results(frames, titles, figsize=(10, 5), cmap=None):
    plt.figure(figsize=figsize)
    for i, (frame, title) in enumerate(zip(frames, titles), start=1):
        plt.subplot(1, len(frames), i)
        plt.title(title)
        plt.imshow(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame,
            cmap=cmap,
        )
        plt.axis("off")
    plt.show()


def heatmap(output_image):
    heat_map = np.zeros(output_image.shape[0:2])
    for row in range(output_image.shape[0]):
        for col in range(output_image.shape[1]):
            if (output_image[row][col] == [0, 255, 0]).all():
                # print("Inside loop")
                heat_map[row][col] = 255

    return heat_map


def final_heatmap(heat_map, segmentation):
    final_heatmap = np.zeros(heat_map.shape)
    for row in range(heat_map.shape[0]):
        if row >= segmentation.shape[0]:
            continue
        for col in range(heat_map.shape[1]):
            if col >= segmentation.shape[1]:
                continue
            if heat_map[row, col] == 255.0:

                final_heatmap[row][col] = (
                    segmentation[row][col] * 0.5 + heat_map[row, col] * 0.5
                )

    return final_heatmap


def optical_flow_heatmap(frame1, frame2):
    tile_width = 4
    tile_height = 4
    points1, points2 = feature_matching_and_flow(frame1, frame2)
    flow_image = draw_optical_flow_vectors(frame1.copy(), points1, points2)
    flow = calculate_dense_optical_flow(frame1, frame2)
    flat_areas = analyze_optical_flow(flow, 4, 4)
    output_image = outputimage(flat_areas, frame2, tile_width, tile_height)
    heat_map = heatmap(output_image)

    display_results(
        [frame2, flow_image, heat_map],
        ["Original", "Optical Flow Vectors", "Optical Flow HeatMap"],
    )
    return heat_map


def final_combined_heatmap(heat_map, segmentation):

    final = final_heatmap(heat_map, segmentation)
    display_results([final], ["Final heatmap"])
    return final
