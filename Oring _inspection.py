import cv2 as cv
import numpy as np
import time
import os


# histogram function - counts how many pixels there are at each grey level
# from lecture 2 slide 5: "An image histogram shows us the distribution of grey levels"
# going through every pixel and incrementing the right bin
def compute_histogram(grey_img):
    hist = np.zeros(256, dtype=np.int64)
    rows = grey_img.shape[0]
    cols = grey_img.shape[1]
    for r in range(rows):
        for c in range(cols):
            gl = grey_img[r, c]
            hist[gl] += 1
    return hist


# threshold using the clustering algorithm from lecture 2 slide 10
# 1. start with T = mean grey level
# 2. split into C1 (> T) and C2 (<= T)
# 3. get mean of each cluster (mu1, mu2)
# 4. new T = (mu1 + mu2) / 2
# 5. repeat until T doesnt change
def find_threshold(grey_img):
    T = int(np.mean(grey_img))

    while True:
        above = grey_img[grey_img > T]
        below = grey_img[grey_img <= T]

        # edge case - if one cluster is empty just stop
        if len(above) == 0 or len(below) == 0:
            break

        mu1 = np.mean(above)
        mu2 = np.mean(below)

        new_T = int((mu1 + mu2) / 2)

        if new_T == T:
            break

        T = new_T

    return T


# thresholding - lecture 2 slide 4
# "convert a grey level image into a binary image depending on whether
#  the pixels are above or below a threshold t"
# pixel > T -> white (255), else -> black (0)
def threshold_image(grey_img, T):
    rows = grey_img.shape[0]
    cols = grey_img.shape[1]
    binary = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if grey_img[r, c] > T:
                binary[r, c] = 255
    return binary


# dilation - lecture 2 slide 27
# "if at least one pixel in the structuring element coincides with a
#  foreground pixel in the image underneath then the input pixel is set
#  to the foreground value"
def dilation(binary, kernel):
    ksize = kernel.shape[0]
    pad   = ksize // 2
    rows  = binary.shape[0]
    cols  = binary.shape[1]
    out   = np.zeros((rows, cols), dtype=np.uint8)

    for r in range(pad, rows - pad):
        for c in range(pad, cols - pad):
            hit = False
            for i in range(ksize):
                for j in range(ksize):
                    if kernel[i, j] == 1:
                        if binary[r - pad + i, c - pad + j] == 255:
                            hit = True
            if hit:
                out[r, c] = 255

    return out


# erosion - lecture 2 slide 26
# "if every pixel in the structuring element corresponds with the image
#  pixels underneath then the input pixel is left as it is"
def erosion(binary, kernel):
    ksize = kernel.shape[0]
    pad   = ksize // 2
    rows  = binary.shape[0]
    cols  = binary.shape[1]
    out   = np.zeros((rows, cols), dtype=np.uint8)

    for r in range(pad, rows - pad):
        for c in range(pad, cols - pad):
            all_fg = True
            for i in range(ksize):
                for j in range(ksize):
                    if kernel[i, j] == 1:
                        if binary[r - pad + i, c - pad + j] == 0:
                            all_fg = False
            if all_fg:
                out[r, c] = 255

    return out


# closing = dilation then erosion - lecture 2 slide 28
# "by performing an erosion on the image after the dilation i.e. a closing
#  we reduce some of this effect. the size of the holes you fill depends
#  on the structuring element"
def closing(binary, kernel):
    return erosion(dilation(binary, kernel), kernel)


# connected component labelling - queue-based approach from lecture 2 slides 32-33
# algorithm:
#  set curlab = 1
#  scan pixel by pixel - if foreground and unlabelled give it curlab, add to queue
#  pop from queue, check all 8 neighbours (8-connectivity)
#  if neighbour is foreground and unlabelled give it curlab, push to queue
#  when queue empty increment curlab, keep scanning
def connected_components(binary):
    rows   = binary.shape[0]
    cols   = binary.shape[1]
    labels = np.zeros((rows, cols), dtype=np.int32)
    curlab = 1

    for r in range(rows):
        for c in range(cols):
            if binary[r, c] == 255 and labels[r, c] == 0:
                labels[r, c] = curlab
                queue = [(r, c)]

                while len(queue) > 0:
                    pr, pc = queue.pop(0)

                    # 8-connected neighbours
                    neighbours = [
                        (pr - 1, pc - 1), (pr - 1, pc), (pr - 1, pc + 1),
                        (pr,     pc - 1),                (pr,     pc + 1),
                        (pr + 1, pc - 1), (pr + 1, pc), (pr + 1, pc + 1)
                    ]

                    for nr, nc in neighbours:
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if binary[nr, nc] == 255 and labels[nr, nc] == 0:
                                labels[nr, nc] = curlab
                                queue.append((nr, nc))

                curlab += 1

    return labels


# find the biggest region - that will be the oring
# lecture 2 slide 35: "Area - the number of pixels in a region"
def find_largest_region(labels):
    best_label = -1
    best_count = 0

    num_labels = labels.max()
    for lbl in range(1, num_labels + 1):
        count = np.sum(labels == lbl)
        if count > best_count:
            best_count = count
            best_label = lbl

    return best_label


# region properties - lecture 2 slides 35-36
# calculates: area, centroid, bounding box, circularity, fill ratio
#
# circularity from slide 36:
#   circularity = mean_dist / (std_dist + 1)
#   where mean_dist and std_dist are the mean and std of distances
#   from the centroid to the perimeter pixels
#
# fill ratio = area / bounding box area
#   a good thick oring fills most of its bounding box
#   a broken or thin oring fills less
def region_properties(labels, target):
    area     = 0
    sum_r    = 0
    sum_c    = 0

    min_r = labels.shape[0]
    max_r = 0
    min_c = labels.shape[1]
    max_c = 0

    rows = labels.shape[0]
    cols = labels.shape[1]

    for r in range(rows):
        for c in range(cols):
            if labels[r, c] == target:
                area   += 1
                sum_r  += r
                sum_c  += c
                if r < min_r: min_r = r
                if r > max_r: max_r = r
                if c < min_c: min_c = c
                if c > max_c: max_c = c

    centroid_r = sum_r / area
    centroid_c = sum_c / area

    # find perimeter pixels - a pixel is on the perimeter if at least one
    # of its 4-connected neighbours doesnt have the same label
    distances = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if labels[r, c] == target:
                up    = labels[r - 1, c]
                down  = labels[r + 1, c]
                left  = labels[r, c - 1]
                right = labels[r, c + 1]
                if up != target or down != target or left != target or right != target:
                    d = np.sqrt((r - centroid_r) ** 2 + (c - centroid_c) ** 2)
                    distances.append(d)

    if len(distances) > 0:
        mean_d  = np.mean(distances)
        std_d   = np.std(distances)
        circ    = mean_d / (std_d + 1)
    else:
        circ = 0.0

    bbox_area = (max_r - min_r) * (max_c - min_c)
    if bbox_area > 0:
        fill = area / bbox_area
    else:
        fill = 0.0

    return area, centroid_r, centroid_c, min_r, max_r, min_c, max_c, circ, fill


# classification - lecture 2 slide 38
# "check the region properties against upper and lower bounds that are
#  acceptable for that class of object - this will normally involve
#  conditional if statements"
#
# fill_ratio is the main check:
#   a good oring is thick so it fills more of its bounding box (ratio > 0.38)
#   a damaged/thin oring fills less of the box (ratio < 0.38)
# also check area is reasonable - too small or too large means something is wrong
def classify_oring(area, circularity, fill_ratio):
    result = "PASS"

    if fill_ratio < 0.38:
        result = "FAIL"

    if area < 4000:
        result = "FAIL"

    if area > 13000:
        result = "FAIL"

    return result


# ---- main loop ----
# goes through every image in the Orings folder and runs all the steps

image_folder = "Orings"
image_files  = sorted(os.listdir(image_folder))

for filename in image_files:

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    filepath   = os.path.join(image_folder, filename)
    img_colour = cv.imread(filepath)
    img_grey   = cv.imread(filepath, 0)

    print("Processing: " + filename)

    t_start = time.time()

    # step 1 - histogram
    hist = compute_histogram(img_grey)

    # step 2 - find threshold using clustering algorithm from lecture 2
    T = find_threshold(img_grey)
    print("  Threshold = " + str(T))

    # step 3 - threshold the image
    binary = threshold_image(img_grey, T)

    # make sure the oring is white - if more than half the image is white
    # then we have it backwards so invert
    if np.sum(binary == 255) > (binary.size * 0.5):
        binary = 255 - binary

    # step 4 - closing to fill any holes in the oring region (lecture 2 slide 28)
    kernel       = np.ones((5, 5), dtype=np.uint8)
    binary_clean = closing(binary, kernel)

    # step 5 - connected component labelling (lecture 2 slides 32-33)
    labels = connected_components(binary_clean)

    ring_label = find_largest_region(labels)

    if ring_label == -1:
        print("  No oring found in this image")
        print("")
        continue

    # step 6 - region properties and classification (lecture 2 slides 35-36, 38)
    area, c_r, c_c, mn_r, mx_r, mn_c, mx_c, circ, fill = region_properties(labels, ring_label)

    result = classify_oring(area, circ, fill)

    t_end   = time.time()
    elapsed = t_end - t_start

    print("  Area        = " + str(area))
    print("  Circularity = " + str(round(circ, 3)))
    print("  Fill Ratio  = " + str(round(fill, 3)))
    print("  Result      = " + result)
    print("  Time        = " + str(round(elapsed, 3)) + "s")
    print("")

    # annotate output image using opencv (allowed as per assignment spec)
    output = img_colour.copy()

    if result == "PASS":
        rect_col = (0, 220, 0)
        text_col = (0, 200, 0)
    else:
        rect_col = (0, 0, 220)
        text_col = (0, 0, 200)

    # bounding box around the oring region
    cv.rectangle(output, (mn_c, mn_r), (mx_c, mx_r), rect_col, 2)

    # pass/fail text
    cv.putText(output, result, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, text_col, 3)

    # processing time on the image (required by assignment)
    cv.putText(output, "Time: " + str(round(elapsed, 3)) + "s",
               (10, 72), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # extra info for debugging
    cv.putText(output, "Fill: " + str(round(fill, 3)),
               (10, 96), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.putText(output, "Circ: " + str(round(circ, 3)),
               (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # filename at the bottom
    cv.putText(output, filename, (10, output.shape[0] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv.imshow("Oring: " + filename, output)
    cv.waitKey(0)
    cv.destroyAllWindows()
