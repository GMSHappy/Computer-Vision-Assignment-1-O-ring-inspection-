import cv2 as cv
import numpy as np
import time
import os



def compute_histogram(grey_img):

    # imagine 256 empty buckets, one for each shade (0 = black, 255 = white)
    hist = np.zeros(256, dtype=np.int64)
    
    # find out how tall and wide the image is
    rows = grey_img.shape[0]
    cols = grey_img.shape[1]
    
    # visit every pixel in the image, one by one 
    for r in range(rows):
        for c in range(cols):

            # check what shade of grey this pixel is (0-255)
            gl = grey_img[r, c]
        
            hist[gl] += 1
    
    return hist

def find_threshold(grey_img):

    # add up all pixel shades and divide by the number of pixels = average shade
    # this becomes our "dividing line" between dark and light
    # e.g. average shade is 120? anything below 120 = black, above 120 = white

 # start with the average shade as our first guess for the dividing line
    T = int(np.mean(grey_img))

    while True:

        above = grey_img[grey_img > T]   # pixels brighter than T (the "light" group)
        below = grey_img[grey_img <= T]  # pixels darker than T  (the "dark" group)

        # safety check - if one group is empty, we cant improve T anymore so stop
        if len(above) == 0 or len(below) == 0:
            break

        # find the average shade of each group
        mu1 = np.mean(above)  # average of the light pixels
        mu2 = np.mean(below)  # average of the dark pixels

        # our new dividing line = halfway between the two averages
        new_T = int((mu1 + mu2) / 2)

        # if T didnt change, we've found the best dividing line so stop
        if new_T == T:
            break
        T = new_T

    return T

def threshold_image(grey_img, T):
    rows = grey_img.shape[0]
    cols = grey_img.shape[1]
    
    # create a blank black image the same size as the original
    binary = np.zeros((rows, cols), dtype=np.uint8)
    
    # visit every pixel
    for r in range(rows):
        for c in range(cols):
            # if this pixel is brighter than our dividing line, make it white
            # otherwise it stays black (already set to 0 above)
            if grey_img[r, c] > T:
                binary[r, c] = 255
    
    # return the new black and white image
    return binary


def dilation(binary, kernel):
    ksize = kernel.shape[0]  # size of the kernel (e.g. 3 = a 3x3 grid)
    pad   = ksize // 2       # how many pixels to stay away from the edge (e.g. 1 for 3x3)
    rows  = binary.shape[0]
    cols  = binary.shape[1]
    out   = np.zeros((rows, cols), dtype=np.uint8)  # blank output image

    # visit every pixel that isnt on the border
    for r in range(pad, rows - pad):
        for c in range(pad, cols - pad):
            hit = False
            
            # slide the kernel over this pixel and check its neighbours
            for i in range(ksize):
                for j in range(ksize):
                    # only check spots where the kernel has a 1
                    if kernel[i, j] == 1:
                        # if ANY neighbour is white, this pixel becomes white
                        if binary[r - pad + i, c - pad + j] == 255:
                            hit = True
            if hit:
                out[r, c] = 255

    # dilation GROWS white regions - if you touch white, you become white
    return out


def erosion(binary, kernel):
    ksize = kernel.shape[0]  # size of the kernel
    pad   = ksize // 2       # border buffer
    rows  = binary.shape[0]
    cols  = binary.shape[1]
    out   = np.zeros((rows, cols), dtype=np.uint8)  # blank output image

    # visit every pixel that isnt on the border
    for r in range(pad, rows - pad):
        for c in range(pad, cols - pad):
            all_fg = True
            
            # slide the kernel over this pixel and check its neighbours
            for i in range(ksize):
                for j in range(ksize):
                    # only check spots where the kernel has a 1
                    if kernel[i, j] == 1:
                        # if ANY neighbour is black, this pixel becomes black
                        if binary[r - pad + i, c - pad + j] == 0:
                            all_fg = False
            if all_fg:
                out[r, c] = 255

    # erosion SHRINKS white regions - if you touch black, you become black
    return out


def closing(binary, kernel):
    # step 1: dilate  (grow white regions - fills in small holes/gaps)
    # step 2: erode   (shrink back down - restores original size)
    # the result: holes inside white regions get filled in
    return erosion(dilation(binary, kernel), kernel)


def connected_components(binary):
    rows   = binary.shape[0]
    cols   = binary.shape[1]
    
    # create a blank label map - each white blob will get its own number
    labels = np.zeros((rows, cols), dtype=np.int32)
    curlab = 1  # start labelling blobs from 1

    for r in range(rows):
        for c in range(cols):
            # if this pixel is white and hasnt been labelled yet, its a new blob
            if binary[r, c] == 255 and labels[r, c] == 0:
                labels[r, c] = curlab
                queue = [(r, c)]  # add it to the queue to explore its neighbours

                # keep going until weve explored the whole blob
                while len(queue) > 0:
                    pr, pc = queue.pop(0)  # grab the next pixel to explore

                    # check all 8 surrounding neighbours (including diagonals)
                    neighbours = [
                        (pr - 1, pc - 1), (pr - 1, pc), (pr - 1, pc + 1),
                        (pr,     pc - 1),                (pr,     pc + 1),
                        (pr + 1, pc - 1), (pr + 1, pc), (pr + 1, pc + 1)
                    ]

                    for nr, nc in neighbours:
                        # make sure the neighbour is inside the image
                        if 0 <= nr < rows and 0 <= nc < cols:
                            # if its white and unlabelled, its part of this blob
                            if binary[nr, nc] == 255 and labels[nr, nc] == 0:
                                labels[nr, nc] = curlab  # give it the same label
                                queue.append((nr, nc))   # add it to explore later

                curlab += 1  # finished this blob, next blob gets the next number

    # return the label map - every pixel now has a number showing which blob it belongs to
    return labels

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

def classify_oring(area, circularity, fill_ratio):

    result = "PASS"

    if fill_ratio < 0.38:
        result = "FAIL"

    if area < 4000:
        result = "FAIL"

    if area > 13000:
        result = "FAIL"

    return result

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


    hist = compute_histogram(img_grey)

    T = find_threshold(img_grey)
    print("  Threshold = " + str(T))

  
    binary = threshold_image(img_grey, T)

    if np.sum(binary == 255) > (binary.size * 0.5):
        binary = 255 - binary

   
    kernel       = np.ones((5, 5), dtype=np.uint8)
    binary_clean = closing(binary, kernel)

  
    labels = connected_components(binary_clean)

    ring_label = find_largest_region(labels)

    if ring_label == -1:
        print("  No oring found in this image")
        print("")
        continue

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


    output = img_colour.copy()

    if result == "PASS":
        rect_col = (0, 220, 0)
        text_col = (0, 200, 0)
    else:
        rect_col = (0, 0, 220)
        text_col = (0, 0, 200)


    cv.rectangle(output, (mn_c, mn_r), (mx_c, mx_r), rect_col, 2)

   
    cv.putText(output, result, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, text_col, 3)


    cv.putText(output, "Time: " + str(round(elapsed, 3)) + "s",
               (10, 72), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.putText(output, "Fill: " + str(round(fill, 3)),
               (10, 96), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.putText(output, "Circ: " + str(round(circ, 3)),
               (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.putText(output, filename, (10, output.shape[0] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv.imshow("Oring: " + filename, output)
    cv.waitKey(0)
    cv.destroyAllWindows()