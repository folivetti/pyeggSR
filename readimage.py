import cv2 
import zipfile
from roifile import ImagejRoi
import numpy as np 
import os 

TRAIN_SCALE = 0.1
TEST_SCALE = 0.1

def _read_inputs(input_filename, isTest=False):
    scale = TEST_SCALE if isTest else TRAIN_SCALE

    in_image_origin = cv2.imread(input_filename)
    in_image_origin_size = in_image_origin.shape

    in_image_origin_hsv = cv2.resize(cv2.cvtColor(in_image_origin.copy(), cv2.COLOR_BGR2HSV), (0, 0), fx=scale, fy=scale)

    in_image_origin = cv2.resize(in_image_origin, (0, 0), fx=scale, fy=scale)
    new_image_in = [in_image_origin[:,:,i] for i in range(in_image_origin.shape[2])]
    new_image_in_hsv = [in_image_origin_hsv[:,:,i] for i in range(in_image_origin_hsv.shape[2])]
    return new_image_in + new_image_in_hsv, in_image_origin_size
        
            
def _read_mask(roi_filename, img_size, isTest=False):
    scale = TEST_SCALE if isTest else TRAIN_SCALE
    mask = np.zeros(img_size)
    if roi_filename[-4:]=='.roi':
        polygons = read_polygons_from_roi(roi_filename)
        fill_polygons_as_labels(mask, polygons)
    elif roi_filename[-4:]=='.zip':
        zip = zipfile.ZipFile(roi_filename)
        zip.extractall(roi_filename[:-4])
        for f in os.listdir(roi_filename[:-4]):
            polygons = read_polygons_from_roi(roi_filename[:-4]+'/'+f)
            fill_polygons_as_labels(mask, polygons)
    elif roi_filename[-4:]=='.png':
        mask = cv2.imread(roi_filename)

    if mask.max() <= 1.0:
        mask = mask * 255
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (0,0), fx=scale, fy=scale)[:,:,0]
    return mask

def read_polygons_from_roi(filename, scale=1.0):
    #print(filename)
    rois = ImagejRoi.fromfile(filename)
    #print(rois)
    if type(rois) == ImagejRoi:
        return [rois.coordinates()]
    polygons = [roi.coordinates() for roi in rois]
    return polygons

def fill_polygons_as_labels(mask, polygons):
    for i, polygon in enumerate(polygons):
        cv2.fillPoly(mask, pts=np.int32([polygon]), color=i + 1)
    return mask
