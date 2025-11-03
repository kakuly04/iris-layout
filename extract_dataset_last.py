import argparse
import logging

import cv2
import numpy as np
from numpy.fft import fft2, ifft2
from math import exp, log, pow
import json
import pickle
import re

# Derived from reference code generated as follows:
#   Prompt: "give me an example implementation of using a fourier-mellin transform to correct for rotation and scale between two images"
#   Model: ChatGPT 4o

def log_polar_transform(image):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    max_radius = min(center[0], center[1])
    log_base = max_radius / np.log(max_radius)
    return cv2.logPolar(image, center, log_base, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

def phase_correlation(image1, image2):
    f1 = fft2(image1)
    f2 = fft2(image2)
    cross_power_spectrum = (f1 * np.conj(f2)) / np.abs(f1 * np.conj(f2))
    inverse_fft = ifft2(cross_power_spectrum)
    return np.abs(inverse_fft)

def find_rotation_and_scale(image1, image2):
    # Apply log-polar transformation to convert rotation and scaling to translation
    image1_logpolar = log_polar_transform(image1)
    image2_logpolar = log_polar_transform(image2)

    # Compute the phase correlation to find translation (rotation and scale)
    result = phase_correlation(image1_logpolar, image2_logpolar)

    if False:
        cv2.imshow("img1 refpolar", image1_logpolar)
        cv2.imshow("img2 refpolar", image2_logpolar)
        cv2.imshow("phase_correlation", cv2.normalize(result, 0, 255, cv2.NORM_MINMAX))
        cv2.waitKey(0)

    # Find the peak of the correlation to get rotation and scaling difference
    while True:
        max_loc = np.unravel_index(np.argmax(result), result.shape)

        # Compute scale and rotation
        rotation_angle = (max_loc[0] - image1.shape[0] / 2) * (360.0 / image1.shape[0])

        # from https://github.com/sthoduka/imreg_fmt/tree/master
        rows = image1.shape[0] # height
        cols = image1.shape[1] # width
        logbase = exp(log(rows * 1.1 / 2.0) / max(rows, cols))

        scale = pow(logbase, max_loc[1])
        scale = 1.0 / scale

        if scale > 0.5 and scale < 2.0:
            return rotation_angle, scale
        else:
            # eliminate the point and try the next smallest
            arr_copy = result.copy()
            # Replace largest values with -inf
            arr_copy[max_loc] = -np.inf
            result = arr_copy

def correct_rotation_and_scale(image, rotation_angle, scale_factor):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    # Correct for rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    # Correct for scale
    corrected_image = cv2.resize(rotated_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    return corrected_image

def snap_to_max(img, max_dim):
    snap = np.zeros((max_dim, max_dim), np.uint8)
    offset_x = (snap.shape[1] - img.shape[1]) // 2
    offset_y = (snap.shape[0] - img.shape[0]) // 2
    snap[offset_y:offset_y + img.shape[0], offset_x:offset_x+img.shape[1]] = img
    return snap

def snap_to_max_rgb(img, max_dim):
    snap = np.zeros((max_dim, max_dim, 3), np.uint8)
    offset_x = (snap.shape[1] - img.shape[1]) // 2
    offset_y = (snap.shape[0] - img.shape[0]) // 2
    snap[offset_y:offset_y + img.shape[0], offset_x:offset_x+img.shape[1], :] = img
    return snap

def map_name_to_celltype(cell_name):
    cell_name = cell_name.lower()
    cell_match = re.search('__(.*)', cell_name)
    if cell_match is None:
        return 'other'
    nm = cell_match.group(1)

    if nm.startswith('xor') or nm.startswith('xnor'):
        return 'logic'
    elif nm.startswith('sed') or nm.startswith('sd') or nm.startswith('df') or nm.startswith('edf'):
        return 'ff'
    elif nm.startswith('dly'):
        return 'logic'
    elif nm.startswith('dl'): # this must be after 'dly'
        return 'ff'
    elif nm.startswith('or') or nm.startswith('nor'):
        return 'logic'
    elif nm.startswith('and') or nm.startswith('nand'):
        return 'logic'
    elif nm.startswith('mux'):
        return 'logic'
    elif nm.startswith('inv') or nm.startswith('einv'):
        return 'logic'
    elif nm.startswith('buf') or nm.startswith('ebuf'):
        return 'logic'
    elif nm.startswith('fa'):
        return 'logic'
    elif nm.startswith('mux'):
        return 'logic'
    elif nm.startswith('clk'):
        return 'logic'
    elif nm.startswith('a'): # this is last in the if/else chain so more specific patterns evaluate first
        return 'logic'
    elif nm.startswith('o'): # this is last in the if/else chain so more specific patterns evaluate first
        return 'logic'
    elif nm.startswith('decap'):
        return 'fill'
    else:
        return 'other'

def reduced_types():
    return ['ff', 'logic', 'fill']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CNN training datasets from images and GDS design data")
    parser.add_argument(
        "--loglevel", required=False, help="set logging level (INFO/DEBUG/WARNING/ERROR)", type=str, default="INFO",
    )
    parser.add_argument(
        "--psi90", action="store_true", help="Look for 90 degree rotated versions of blocks"
    )
    parser.add_argument(
        "--layer", required=False, help="Layer to process", choices=['poly', 'm1'], default='poly'
    )
    parser.add_argument(
        "--tech", required=False, help="Tech library", choices=['sky130'], default='sky130'
    )
    
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    names = ["wb_bridge_2way", "wrapped_etpu", "wb_openram_wrapper", "wrapped_function_generator", "wrapped_ibnalhaytham", "wrapped_mbsFSK", "wrapped_silife", "wrapped_snn_network", "housekeeping"]
    #names = args.names
    #names = ["wb_openram_wrapper"]

    layer = args.layer
    tech = args.tech

    if args.psi90:
        try_angles = ['', '_psi90']
    else:
        try_angles = ['']

    for psi in try_angles:
        for name in names:
            print(f"Processing {name} layer {layer} angle {psi}")
            # Load two images (they should be grayscale for simplicity)
            try:
                gds_png = cv2.imread(f"imaging/{name}_{layer}.png", cv2.IMREAD_GRAYSCALE)
                image = cv2.imread(f"cropped_image/{name}_psi.png", cv2.IMREAD_GRAYSCALE)
                gds_label_image = cv2.imread(f"imaging/{name}_{layer}_label.png", cv2.IMREAD_UNCHANGED)

                if '_psi90' in psi:
                    gds_png = 255 - gds_png
            except:
                print(f"Rotated version {name} not found, skipping")
                continue

            if image is None:
                print(f"Rotated version {name} not found, skipping")
                continue

            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            max_dim = max(max(gds_png.shape), max(image.shape))
            gds_png_snap = snap_to_max(gds_png, max_dim)
            image_snap = snap_to_max(image, max_dim)
            gds_label_image_snap = snap_to_max_rgb(gds_label_image, max_dim)

            #cv2.imshow("GDS Snap", gds_png_snap)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #cv2.imshow("Image Snap", image_snap) 
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    
            # Find rotation and scale difference
            
            rotation_angle, scale_factor = find_rotation_and_scale(gds_png_snap, image_snap)
            print(f"Rotation angle: {rotation_angle}, Scale factor: {scale_factor}")

            # Correct img2 for rotation and scale
            corrected_image = correct_rotation_and_scale(image, 0, scale_factor)
            corrected_gds = correct_rotation_and_scale(gds_png_snap, 0, scale_factor)
            corrected_gds_label = correct_rotation_and_scale(gds_label_image_snap, 0, scale_factor)

            print("Corrected Image Shape:", corrected_image.shape)
            print("GDS PNG Shape:", gds_png.shape)
            print("GDS PNG Snap Shape:", gds_png_snap.shape)
            print("Image Shape:", image.shape)
            print("Image Snap Shape:", image_snap.shape)

            
            if corrected_image.shape[0] >= gds_png.shape[0] and corrected_image.shape[1] >= gds_png.shape[1]:
            # now template match the reference onto the image so we can determine the offset
                corr = cv2.matchTemplate(corrected_image, gds_png, cv2.TM_CCOEFF)
                _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(corr)
                # create the composite
                print("Max Location:", max_loc)
                composite_overlay = np.zeros(corrected_image.shape, np.uint8)
                composite_overlay[max_loc[1]:max_loc[1] + gds_png.shape[0], max_loc[0]: max_loc[0] + gds_png.shape[1]] = gds_png
                cv2.imwrite(f'imaging/{name}_{layer}_corrected_2.png', corrected_image)
                blended = cv2.addWeighted(corrected_image, 1.0, composite_overlay, 0.5, 0)
                cv2.imwrite(f'imaging/{name}_{layer}_blended_2.png', blended)
            
            else:
                print("Corrected image is smaller than or approximately the same size than the image")
                corrected_image_corrected = correct_rotation_and_scale(image_snap, 0, scale_factor)
                corr = cv2.matchTemplate(corrected_image_corrected, gds_png, cv2.TM_CCOEFF)
                _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(corr)
                # create the composite
                print("Max Location:", max_loc)
                composite_overlay = np.zeros(corrected_image_corrected.shape, np.uint8)
                composite_overlay[max_loc[1]:max_loc[1] + gds_png.shape[0], max_loc[0]: max_loc[0] + gds_png.shape[1]] = gds_png
                cv2.imwrite(f'imaging/{name}_{layer}_corrected_2.png', corrected_image_corrected)
                blended = cv2.addWeighted(corrected_image_corrected, 1.0, composite_overlay, 0.5, 0)
                #max_loc = (0, 0)
                #print("Max Location:", max_loc)
                cv2.imwrite(f'imaging/{name}_{layer}_blended_2.png', blended)
            

            #composite_overlay = corrected_gds
            #composite_overlay[max_loc[1]:max_loc[1] + gds_png.shape[0], max_loc[0]: max_loc[0] + gds_png.shape[1]] = gds_png
            
            
            #cv2.imshow("Composite Overlay", composite_overlay)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            if corrected_image.shape[0] >= gds_png.shape[0] and corrected_image.shape[1] >= gds_png.shape[1]:
                corrected_image_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2BGR)
                label_overlay = np.zeros(corrected_image_rgb.shape, np.uint8)
                label_overlay[max_loc[1]:max_loc[1] + gds_label_image.shape[0], max_loc[0]: max_loc[0] + gds_label_image.shape[1], :] = gds_label_image
            else:
                corrected_image_rgb = cv2.cvtColor(corrected_image_corrected, cv2.COLOR_GRAY2BGR)
                label_overlay = np.zeros(corrected_image_rgb.shape, np.uint8)
                #label_overlay = corrected_gds_label
                label_overlay[max_loc[1]:max_loc[1] + gds_label_image.shape[0], max_loc[0]: max_loc[0] + gds_label_image.shape[1], :] = gds_label_image
                corrected_image = corrected_image_corrected
            
            #label_overlay = corrected_gds_label
            #label_overlay[max_loc[1]:max_loc[1] + gds_label_image.shape[0], max_loc[0]: max_loc[0] + gds_label_image.shape[1], :] = gds_label_image
            label_blended = cv2.addWeighted(corrected_image_rgb, 1.0, label_overlay, 0.3, 0)  
            cv2.imwrite(f'imaging/{name}_{layer}_aligned_2.png', label_blended)
            
            #cv2.imshow("Label_Blended", label_blended)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            with open(f'imaging/{args.tech}_cells.json', 'r') as f:
                cell_names = json.load(f)

            with open(f'imaging/{name}_{layer}_lib.json', 'r') as f:
                cells = json.load(f)

            # check alignment by drawing the rectangles
            cell_overlay = np.zeros(corrected_image_rgb.shape, np.uint8)
            max_x = 0
            max_y = 0
            entry = {}
            entry['labels'] = []
            entry['data'] = []
            labels = {} # this is no longer used, we pull the label index from the master label list
            label_count = 0
            for cell in cells.values():
                if cell[2] not in labels:
                    labels[cell[2]] = label_count
                    label_count += 1
                cv2.rectangle(
                    cell_overlay,
                    [cell[0][0][0] + max_loc[0], cell[0][0][1] + max_loc[1]],
                    [cell[0][1][0] + max_loc[0], cell[0][1][1] + max_loc[1]],
                    cell[1],
                    thickness=-1
                )
                # this is used to sanity check the statically coded dimensions of the x/y image crops
                if abs(cell[0][0][0] - cell[0][1][0]) > max_x:
                    max_x = abs(cell[0][0][0] - cell[0][1][0])
                if abs(cell[0][0][1] - cell[0][1][1]) > max_y:
                    max_y = abs(cell[0][0][1] - cell[0][1][1])

                # extract a rectangle around the center of each standard cell and save it in a labelled training set
                data = np.zeros((32, 64, 3), dtype=np.uint8)
                center_x = (cell[0][0][0] + cell[0][1][0]) // 2 + max_loc[0]
                center_y = (cell[0][0][1] + cell[0][1][1]) // 2 + max_loc[1]
                if center_y - 16 >= 0 and center_y + 16 < corrected_image.shape[0] \
                and center_x - 32 >= 0 and center_x + 32 < corrected_image.shape[1]:
                    data = cv2.cvtColor(corrected_image[center_y - 16:center_y + 16, center_x - 32:center_x + 32], cv2.COLOR_GRAY2RGB)
                    try:
                        reduced_name = map_name_to_celltype(cell[2])
                    except ValueError:
                        print(f'Cell not in master cell list: {cell[2]}; skipping')
                        continue
                    if reduced_name != 'other': # throw out the "other" label
                        label_index = reduced_types().index(reduced_name)
                        entry['labels'].append(label_index) # substitute with a numeric value so it can be converted to a tensor
                        entry['data'].append(data)

                        if False: # manual check data extraction
                            blended_rect = cv2.addWeighted(corrected_image, 1.0, cell_overlay, 0.5, 0)
                            cv2.imshow("Rectangles", blended_rect)
                            cv2.imshow('data', data)
                            print(f'{reduced_name}: {label_index}')
                            cv2.waitKey(0)

            # dump the data into pickle files for consumption by downstream CNN pipeline
            print(f'max_x: {max_x}, max_y: {max_y}')
            for i in range(len(reduced_types())):
                print(f"{reduced_types()[i]}: {entry['labels'].count(i)}")
            with open(f'imaging/{name}_{layer}{psi}_2.pkl', 'wb') as f:
                pickle.dump(entry, f)
            meta = {
                'num_cases_per_batch' : len(entry['data']),
                'label_names' : reduced_types(),
                'num_vis' : 64 * 32 * 3,
            }
            with open(f'imaging/{name}_{layer}{psi}_2.meta', 'wb') as f:
                pickle.dump(meta, f)

            # Quality check the alignment
            blended_rect = cv2.addWeighted(corrected_image_rgb, 1.0, cell_overlay, 1, 0)

            #cv2.imshow("Corrected Image", corrected_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #cv2.imshow("Cell Overlay", cell_overlay)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # Display the corrected image
            # cv2.imshow("Corrected Image", corrected_img2)
            # cv2.imshow("Reference image", img1)
            #cv2.imshow("Correlation", cv2.normalize(corr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #cv2.imshow("Blended", blended)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #cv2.imshow("Rectangles", blended_rect)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # Now read in the JSON file with cell locations, and use this to create a training set of data
            # This consists of:
            #  - A monochrome rectangle that defines the region of interest; the color is correlated to the gate type
            #  - The underlying source image, cropped to a fixed size that represents the maximum search window for a
            #    gate of any size (equal to the biggest standard cell plus some alignment margin)
            #  - The representation of the "true gate" as a black and white image, correlated to the source image
            #
            # The input to the classifier would be a source image area, that is the same as the fixed size used in training
            # The output of the classifier is a tensor of potential gate matches, which we will threshold into "most likely match"
