import cv2 as cv
import numpy as np
import time
import os

# ================================================================
# STEP 1 - COMPUTE THE IMAGE HISTOGRAM
# From Lecture 2: "An image histogram shows us the distribution
# of grey levels in an image."
# We count how many pixels exist at each brightness level 0-255
# ================================================================
def compute_histogram(gray_image):

    histogram = np.zeros(256, dtype=np.int64)

    for x in range(0, gray_image.shape[0]):
        for y in range(0, gray_image.shape[1]):
            pixel_value = gray_image[x, y]
            histogram[pixel_value] = histogram[pixel_value] + 1

    return histogram


# ================================================================
# STEP 2 - FIND THE THRESHOLD USING THE CLUSTERING ALGORITHM
# Directly from Lecture 2 slides (the clustering algorithm):
#   1. Select initial T = average grey level of the image
#   2. Split pixels into C1 (> T) and C2 (<= T)
#   3. Compute mean of C1 = mu1, mean of C2 = mu2
#   4. New T = (mu1 + mu2) / 2
#   5. Repeat until T stops changing
# ================================================================
def find_threshold(gray_image):

    T = int(np.mean(gray_image))

    while True:
        C1_pixels = gray_image[gray_image > T]
        C2_pixels = gray_image[gray_image <= T]

        if len(C1_pixels) == 0 or len(C2_pixels) == 0:
            break

        mu1 = np.mean(C1_pixels)
        mu2 = np.mean(C2_pixels)

        T_new = int((mu1 + mu2) / 2)

        if T_new == T:
            break

        T = T_new

    return T


# ================================================================
# STEP 3 - THRESHOLD THE IMAGE
# From Lecture 2: "convert a grey level image into a binary image
# depending on whether the pixels are above or below a threshold t"
# Based on the sample code style from the starter file given to us.
# ================================================================
def threshold_image(gray_image, thresh):

    binary = np.zeros(gray_image.shape, dtype=np.uint8)

    for x in range(0, gray_image.shape[0]):
        for y in range(0, gray_image.shape[1]):
            if gray_image[x, y] > thresh:
                binary[x, y] = 255
            else:
                binary[x, y] = 0

    return binary


# ================================================================
# STEP 4 - BINARY MORPHOLOGY: CLOSING (Dilation then Erosion)
# From Lecture 2: "By performing an erosion on the image after
# the dilation, i.e. a closing, we reduce some of this effect.
# The size of the holes you fill depends on the structuring element"
#
# DILATION rule - if ANY pixel under structuring element is 255
#                 then set the output pixel to 255
# EROSION rule  - if ALL pixels under structuring element are 255
#                 then set the output pixel to 255, otherwise 0
# ================================================================
def dilation(binary, structuring_element):

    se_size = structuring_element.shape[0]
    pad = se_size // 2

    output = np.zeros(binary.shape, dtype=np.uint8)

    for x in range(pad, binary.shape[0] - pad):
        for y in range(pad, binary.shape[1] - pad):

            found_white = False

            for i in range(0, se_size):
                for j in range(0, se_size):
                    if structuring_element[i, j] == 1:
                        if binary[x - pad + i, y - pad + j] == 255:
                            found_white = True

            if found_white == True:
                output[x, y] = 255

    return output


def erosion(binary, structuring_element):

    se_size = structuring_element.shape[0]
    pad = se_size // 2

    output = np.zeros(binary.shape, dtype=np.uint8)

    for x in range(pad, binary.shape[0] - pad):
        for y in range(pad, binary.shape[1] - pad):

            all_white = True

            for i in range(0, se_size):
                for j in range(0, se_size):
                    if structuring_element[i, j] == 1:
                        if binary[x - pad + i, y - pad + j] == 0:
                            all_white = False

            if all_white == True:
                output[x, y] = 255

    return output


def closing(binary, structuring_element):
    dilated = dilation(binary, structuring_element)
    closed  = erosion(dilated, structuring_element)
    return closed


# ================================================================
# STEP 5 - CONNECTED COMPONENT LABELLING
# From Lecture 2 (the queue-based algorithm from slide 33):
#   1. Scan image pixel by pixel
#   2. When you find an unlabelled white pixel, give it curlab
#      and add it to a queue
#   3. Pop from queue, check its 8 neighbours
#   4. Label any unlabelled white neighbours with curlab too
#      and add them to the queue
#   5. When queue is empty, increment curlab and keep scanning
# ================================================================
def connected_component_labelling(binary):

    labels = np.zeros(binary.shape, dtype=np.int32)
    curlab = 1

    for x in range(0, binary.shape[0]):
        for y in range(0, binary.shape[1]):

            if binary[x, y] == 255 and labels[x, y] == 0:

                queue = []
                labels[x, y] = curlab
                queue.append((x, y))

                while len(queue) > 0:

                    pixel = queue.pop(0)
                    px = pixel[0]
                    py = pixel[1]

                    neighbours = [
                        (px - 1, py),
                        (px + 1, py),
                        (px, py - 1),
                        (px, py + 1),
                        (px - 1, py - 1),
                        (px - 1, py + 1),
                        (px + 1, py - 1),
                        (px + 1, py + 1)
                    ]

                    for nx, ny in neighbours:
                        if nx >= 0 and nx < binary.shape[0] and ny >= 0 and ny < binary.shape[1]:
                            if binary[nx, ny] == 255 and labels[nx, ny] == 0:
                                labels[nx, ny] = curlab
                                queue.append((nx, ny))

                curlab = curlab + 1

    return labels


# ================================================================
# STEP 5b - FIND THE LARGEST REGION (the O-ring)
# From Lecture 2: "Area - the number of pixels in a region"
# The O-ring will be the biggest white region in the image
# ================================================================
def find_largest_region(labels):

    largest_label = -1
    largest_count = 0

    for label in range(1, labels.max() + 1):
        count = np.sum(labels == label)
        if count > largest_count:
            largest_count = count
            largest_label = label

    return largest_label


# ================================================================
# STEP 6 - REGION PROPERTIES
# From Lecture 2: "Area, Circularity, Centroid"
#
# Circularity formula from lecture slide 36:
# uses mean and standard deviation of distances from the centroid
# to every pixel in the region
#
#   Area      = total number of pixels belonging to the region
#   Centroid  = average row and average column of all pixels
#   Bounding box = the min and max row and column of the region
# ================================================================
def region_properties(labels, target_label):

    area = 0
    total_row = 0
    total_col = 0

    min_row = labels.shape[0]
    max_row = 0
    min_col = labels.shape[1]
    max_col = 0

    for x in range(0, labels.shape[0]):
        for y in range(0, labels.shape[1]):
            if labels[x, y] == target_label:
                area = area + 1
                total_row = total_row + x
                total_col = total_col + y

                if x < min_row:
                    min_row = x
                if x > max_row:
                    max_row = x
                if y < min_col:
                    min_col = y
                if y > max_col:
                    max_col = y

    centroid_row = total_row / area
    centroid_col = total_col / area

    total_dist = 0
    distances = []

    for x in range(0, labels.shape[0]):
        for y in range(0, labels.shape[1]):
            if labels[x, y] == target_label:
                dist = np.sqrt((x - centroid_row)**2 + (y - centroid_col)**2)
                distances.append(dist)
                total_dist = total_dist + dist

    mean_dist = total_dist / area
    std_dist  = np.std(distances)

    if mean_dist > 0:
        circularity = mean_dist / (std_dist + 1)
    else:
        circularity = 0

    return area, centroid_row, centroid_col, min_row, max_row, min_col, max_col, circularity


# ================================================================
# STEP 7 - CLASSIFY THE O-RING AS PASS OR FAIL
# From Lecture 2: "check the region properties against upper and
# lower bounds that are acceptable for that class of object.
# This will normally involve a set of conditional (if) statements"
# ================================================================
def classify_oring(area, circularity):

    result = "PASS"

    if circularity < 7.0:
        result = "FAIL"

    if area < 4000:
        result = "FAIL"

    if area > 10000:
        result = "FAIL"

    return result


# ================================================================
# MAIN PROGRAM
# Loops through every O-ring image in the Orings folder and
# runs all 7 steps on each one then shows the annotated result.
# Assignment requires: time measured and displayed on the image.
# ================================================================
image_folder = 'Orings'

image_files = sorted(os.listdir(image_folder))

for filename in image_files:

    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    path = os.path.join(image_folder, filename)

    img_colour = cv.imread(path)
    img_gray   = cv.imread(path, 0)

    print("Processing: " + filename)

    before = time.time()

    histogram = compute_histogram(img_gray)

    thresh = find_threshold(img_gray)
    print("Threshold found: " + str(thresh))

    binary = threshold_image(img_gray, thresh)

    if np.sum(binary == 255) > binary.size * 0.5:
        binary = 255 - binary

    structuring_element = np.ones((5, 5), dtype=np.uint8)
    binary_clean = closing(binary, structuring_element)

    labels = connected_component_labelling(binary_clean)

    oring_label = find_largest_region(labels)

    area, c_row, c_col, min_r, max_r, min_c, max_c, circularity = region_properties(labels, oring_label)

    result = classify_oring(area, circularity)

    after = time.time()
    elapsed = after - before

    print("Time taken: " + str(elapsed))
    print("Area: " + str(area))
    print("Circularity: " + str(circularity))
    print("Result: " + result)
    print("")

    output = img_colour.copy()

    if result == "PASS":
        box_colour  = (0, 255, 0)
        text_colour = (0, 200, 0)
    else:
        box_colour  = (0, 0, 255)
        text_colour = (0, 0, 220)

    cv.rectangle(output, (min_c, min_r), (max_c, max_r), box_colour, 2)

    cv.putText(output, result, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, text_colour, 3)

    cv.putText(output, "Time: " + str(round(elapsed, 3)) + "s", (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.putText(output, "Circ: " + str(round(circularity, 2)), (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.putText(output, filename, (10, output.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv.imshow("O-Ring: " + filename, output)
    cv.waitKey(0)
    cv.destroyAllWindows()
