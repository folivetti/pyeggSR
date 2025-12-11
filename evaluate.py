import numpy as np
import math
import cv2
from scipy.stats import kurtosis, skew
from skimage.morphology import remove_small_objects, remove_small_holes, thin
import skimage.feature as feature
import mahotas

from expr import *
from egraph import *

SHARPEN_KERNEL = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="int")
ROBERT_CROSS_H_KERNEL = np.array(([0, 1], [-1, 0]), dtype="int")
ROBERT_CROSS_V_KERNEL = np.array(([1, 0], [0, -1]), dtype="int")
OPENCV_MIN_KERNEL_SIZE = 3
OPENCV_MAX_KERNEL_SIZE = 31
OPENCV_KERNEL_RANGE = OPENCV_MAX_KERNEL_SIZE - OPENCV_MIN_KERNEL_SIZE
OPENCV_MIN_INTENSITY = 0
OPENCV_MAX_INTENSITY = 255
OPENCV_INTENSITY_RANGE = OPENCV_MAX_INTENSITY - OPENCV_MIN_INTENSITY

KERNEL_SCALE = OPENCV_KERNEL_RANGE / OPENCV_INTENSITY_RANGE


GABOR_SIGMAS = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
GABOR_THETAS = np.arange(0, 2, step=1.0 / 8) * np.pi
GABOR_LAMBDS = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
GABOR_GAMMAS = np.arange(0.0625, 1.001, step=1.0 / 16)
#GABOR_FILTER_KSIZE = 11

#BINARY_FILL_COLOR = 255

IMAGE_UINT8_POSITIVE: int = 255
IMAGE_UINT8_NEGATIVE: int = 0
IMAGE_UINT8_COLOR_1C: list = [IMAGE_UINT8_POSITIVE]
IMAGE_UINT8_COLOR_3C: list = IMAGE_UINT8_COLOR_1C * 3

# Helper functions
def correct_ksize(param):
    """Ensure kernel size is odd and positive."""
    ksize = min(max(1, int(param)), 31)
    if ksize % 2 == 0:
        ksize += 1
    return ksize

def kernel_from_parameters(const_params):
    """Create a morphological kernel from parameters."""
    if len(const_params) < 2:
        # Default kernel
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    shape_param = const_params[0]
    size_param = max(1, int(const_params[1]))
    if size_param % 2 == 0:
        size_param += 1
    
    # Select kernel shape based on parameter
    if shape_param < 85:
        shape = cv2.MORPH_RECT
    elif shape_param < 170:
        shape = cv2.MORPH_ELLIPSE
    else:
        shape = cv2.MORPH_CROSS
    
    return cv2.getStructuringElement(shape, (size_param, size_param))

def gabor_kernel(ksize, theta, sigma):
    """Create a Gabor kernel."""
    lambda_param = 10.0  # wavelength
    gamma = 0.5  # spatial aspect ratio
    psi = 0  # phase offset
    
    # Adjust theta to be in radians
    theta_rad = theta * np.pi / 180.0
    
    # Create Gabor kernel
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta_rad, lambda_param, gamma, psi)
    return kernel

def compute_iou(y_true, y_pred):
    # MetricIOU
    _y_true = y_true
    _y_pred = y_pred
    _y_pred[_y_pred > 0] = 1
    if np.sum(_y_true) == 0:
        _y_true = 1 - _y_true
        _y_pred = 1 - _y_pred
    intersection = np.logical_and(_y_true, _y_pred)
    union = np.logical_or(_y_true, _y_pred)
    return np.sum(intersection) / np.sum(union)


def evaluate_egraph(root : int, egraph : EGraph, consts = [], data=None):
    """
    Evaluate an expression tree with given data for variables and constant parameters.
    
    Args:
        root: The root e-class ID to evaluate
        egraph: The e-graph to evaluate
        data: Dictionary mapping variable indices to their data (numpy arrays)
        const_params: List of constant parameters for operations that need them
    
    Returns:
        The evaluated result (numpy array for image operations)
    """
    if data is None:
        data = {}
    root = egraph.find(root)
    eclass = egraph.map_class[root]
    e = next(iter(eclass.enodes))
    const_params = []
    
    match e:
        # Binary operations
        case Add(l, r):
            left_val, consts = evaluate_egraph(l, egraph, consts, data)
            right_val, consts = evaluate_egraph(r, egraph, consts, data)
            return f_add([left_val, right_val], const_params), consts
        case Sub(l, r):
            left_val, consts = evaluate_egraph(l, egraph, consts, data)
            right_val, consts = evaluate_egraph(r, egraph, consts, data)
            return f_sub([left_val, right_val], const_params), consts
        case AbsoluteDifference2(l, r):
            left_val, consts = evaluate_egraph(l, egraph, consts, data)
            right_val, consts = evaluate_egraph(r, egraph, consts, data)
            return f_absolute_difference2([left_val, right_val], const_params), consts
        case BitwiseAnd(l, r):
            left_val, consts = evaluate_egraph(l, egraph, consts, data)
            right_val, consts = evaluate_egraph(r, egraph, consts, data)
            return f_bitwise_and([left_val, right_val], const_params), consts
        case BitwiseAndMask(l, r):
            left_val, consts = evaluate_egraph(l, egraph, consts, data)
            right_val, consts = evaluate_egraph(r, egraph, consts, data)
            return f_bitwise_and_mask([left_val, right_val], const_params), consts
        case BitwiseOr(l, r):
            left_val, consts = evaluate_egraph(l, egraph, consts, data)
            right_val, consts = evaluate_egraph(r, egraph, consts, data)
            return f_bitwise_or([left_val, right_val], const_params), consts
        case BitwiseXor(l, r):
            left_val, consts = evaluate_egraph(l, egraph, consts, data)
            right_val, consts = evaluate_egraph(r, egraph, consts, data)
            return f_bitwise_xor([left_val, right_val], const_params), consts
        case Min(l, r):
            left_val, consts = evaluate_egraph(l, egraph, consts, data)
            right_val, consts = evaluate_egraph(r, egraph, consts, data)
            return f_min([left_val, right_val], const_params), consts
        case Max(l, r):
            left_val, consts = evaluate_egraph(l, egraph, consts, data)
            right_val, consts = evaluate_egraph(r, egraph, consts, data)
            return f_max([left_val, right_val], const_params), consts
        case Mean(l, r):
            left_val, consts = evaluate_egraph(l, egraph, consts, data)
            right_val, consts = evaluate_egraph(r, egraph, consts, data)
            return f_mean([left_val, right_val], const_params), consts
        # Unary operations
        case Exp(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_exp([child_val], const_params), consts
        case Log(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_log([child_val], const_params), consts
        case Erode(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_erode([child_val], const_params), consts
        case Dilate(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_dilate([child_val], const_params), consts
        case Open(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_open([child_val], const_params), consts
        case Close(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_close([child_val], const_params), consts
        case MorphGradient(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_morph_gradient([child_val], const_params), consts
        case MorphTopHat(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_morph_top_hat([child_val], const_params), consts
        case MorphBlackHat(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_morph_black_hat([child_val], const_params), consts
        case FillHoles(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_fill_holes([child_val], const_params), consts
        case RemoveSmallHoles(x):
            const_params = consts[:1]
            const = consts[1:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_remove_small_holes([child_val], const_params), consts
        case RemoveSmallObjects(x):
            const_params = consts[:1]
            const = consts[1:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_remove_small_objects([child_val], const_params), consts
        case MedianBlur(x):
            const_params = consts[:1]
            const = consts[1:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_median_blur([child_val], const_params), consts
        case GaussianBlur(x):
            const_params = consts[:1]
            const = consts[1:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_gaussian_blur([child_val], const_params), consts
        case Laplacian(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_laplacian([child_val], const_params), consts
        case Sobel(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_sobel([child_val], const_params), consts
        case RobertCross(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_robert_cross([child_val], const_params), consts
        case Canny(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_canny([child_val], const_params), consts
        case Sharpen(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_sharpen([child_val], const_params), consts
        case Kirsch(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_kirsch([child_val], const_params), consts
        case Embossing(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_embossing([child_val], const_params), consts
        case Pyr(x):
            const_params = consts[:1]
            const = consts[1:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_pyr([child_val], const_params), consts
        case Denoizing(x):
            const_params = consts[:1]
            const = consts[1:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_denoizing([child_val], const_params), consts
        case AbsoluteDifference(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_absolute_difference([child_val], const_params), consts
        case RelativeDifference(x):
            const_params = consts[:1]
            const = consts[1:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_relative_difference([child_val], const_params), consts
        case FluoTopHat(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_fluo_top_hat([child_val], const_params), consts
        case GaborFilter(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_gabor_filter([child_val], const_params), consts
        case DistanceTransform(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_distance_transform([child_val], const_params), consts
        case DistanceTransformAndThresh(x):
            const_params = consts[:1]
            const = consts[1:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_distance_transform_and_thresh([child_val], const_params), consts
        case Threshold(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_threshold([child_val], const_params), consts
        case ThresholdAt1(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_threshold_at_1([child_val], const_params), consts
        case BinaryInRange(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_binary_in_range([child_val], const_params), consts
        case InRange(x):
            const_params = consts[:2]
            const = consts[2:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_in_range([child_val], const_params), consts
        case BitwiseNot(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_bitwise_not([child_val], const_params), consts
        case SquareRoot(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_square_root([child_val], const_params), consts
        case Square(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_square([child_val], const_params), consts
        case ThresholdOtsu(x):
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_threshold_otsu([child_val], const_params), consts
        case ContourArea(x):
            const_params = consts[:1]
            consts = consts[1:]
            child_val, consts = evaluate_egraph(x, egraph, consts, data)
            return f_contour_area([child_val], const_params), consts
        # Leaf nodes
        case Var(idx):
            return data[idx], consts
        case _ as unreachable:
            raise ValueError(f"Unknown expression type: {unreachable}")

def _fill_cnt(image, contours):
	assert (
			len(image.shape) == 3 or len(image.shape) == 2
	), "given image wrong format, shape must be (h, w, c) or (h, w)"
	if len(image.shape) == 3 and image.shape[-1] == 3:
		color = IMAGE_UINT8_COLOR_3C
	elif len(image.shape) == 2:
		color = IMAGE_UINT8_COLOR_1C
	else:
		raise ValueError("Image wrong format, must have 1 or 3 channels")
	selected = -1  # selects all the contours (-1)
	thickness = -1  # fills the contours (-1)
	final_img = cv2.drawContours(image, contours, selected, color, thickness)
	#print(f'{final_img.max()}, {final_img.sum()}, {final_img.mean()}')
	return final_img

def f_contour_area(args, const_params):
	# Contours find
	threshold_scaler = 9 # threshold between 0 and 40*255=10200

	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for c in contours:
		if const_params[0] > 128:
			if cv2.contourArea(c) >= const_params[0]-128:
				final_contours.append(c)
		else:
			if cv2.contourArea(c) <= 128 - const_params[0]:
				final_contours.append(c)

	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]

def f_contour_contrast(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		masked_img = _fill_cnt(image.copy(), [cnt]) * image
		sub_img = masked_img[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i*np.pi/8 for i in range(8)], levels=256)
		contrast = feature.graycoprops(graycom, 'contrast').mean()
		threshold = (128-const_params[1]) / 128
		if threshold > 0:
			if contrast > threshold:
				final_contours.append(cnt)
		else:
			if contrast < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]

def f_contour_dissimilarity(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		sub_img = image[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i*np.pi/8 for i in range(8)], levels=256)
		dissimilarity = feature.graycoprops(graycom, 'dissimilarity').mean()
		threshold = (128-const_params[1]) / 128
		if threshold > 0:
			if dissimilarity > threshold:
				final_contours.append(cnt)
		else:
			if dissimilarity < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]

def f_contour_homogeneity(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		sub_img = image[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i * np.pi / 8 for i in range(8)], levels=256)
		homogeneity = feature.graycoprops(graycom, 'homogeneity').mean()
		threshold = (128 - const_params[1]) / 128
		if threshold > 0:
			if homogeneity > threshold:
				final_contours.append(cnt)
		else:
			if homogeneity < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]

def f_contour_ASM(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		sub_img = image[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i * np.pi / 8 for i in range(8)], levels=256)
		asm = feature.graycoprops(graycom, 'ASM').mean()
		threshold = (128 - const_params[1]) / 128
		if threshold > 0:
			if asm > threshold:
				final_contours.append(cnt)
		else:
			if asm < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]

def f_contour_energy(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		sub_img = image[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i * np.pi / 8 for i in range(8)], levels=256)
		energy = feature.graycoprops(graycom, 'energy').mean()
		threshold = (128 - const_params[1]) / 128
		if threshold > 0:
			if energy > threshold:
				final_contours.append(cnt)
		else:
			if energy < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]


def f_contour_correlation(args, const_params):
	image = args[0].copy()
	method = cv2.CHAIN_APPROX_SIMPLE
	mode = cv2.RETR_LIST
	contours = cv2.findContours(image.copy(), mode, method)[0]
	final_contours = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		sub_img = image[y:y+h, x:x+w]
		graycom = feature.graycomatrix(sub_img, [const_params[0]], [i * np.pi / 8 for i in range(8)], levels=256)
		correlation = feature.graycoprops(graycom, 'correlation').mean()
		threshold = (128 - const_params[1]) / 128
		if threshold > 0:
			if correlation > threshold:
				final_contours.append(cnt)
		else:
			if correlation < -threshold:
				final_contours.append(cnt)
	return cv2.threshold(_fill_cnt(image, final_contours), IMAGE_UINT8_POSITIVE-1, IMAGE_UINT8_POSITIVE, cv2.THRESH_BINARY)[1]


def f_local_binary_pattern(args, const_params):
	point_scaler = 0.2 # max point 51
	radius_scaler = 0.04 # max radius 10
	return feature.local_binary_pattern(args[0].copy(), const_params[0], const_params[1])

def f_add(args, const_params):
	return cv2.add(args[0], args[1])

def f_sub(args, const_params):
	return cv2.subtract(args[0], args[1])

def f_mean(args, const_params):
    return cv2.addWeighted(args[0], 0.5, args[1], 0.5, 0)

def f_bitwise_not(args, const_params):
	return cv2.bitwise_not(args[0])

def f_bitwise_or(args, const_params):
	return cv2.bitwise_or(args[0], args[1])

def f_bitwise_and(args, const_params):
	x = args[0].copy()
	x = x.astype(np.uint8)
	y = args[1].copy()
	y = y.astype(np.uint8)
	return cv2.bitwise_and(x, y)

def f_bitwise_and_mask(args, const_params):
	x = args[0].copy()
	x = x.astype(np.uint8)
	y = args[1].copy()
	y = y.astype(np.uint8)
	return cv2.bitwise_and(x, x, mask=y)

def f_bitwise_xor(args, const_params):
	return cv2.bitwise_xor(args[0], args[1])


def f_square_root(args, const_params):
	return (cv2.sqrt((args[0] / 255.0).astype(np.float32)) * 255).astype(np.uint8)


def f_square(args, const_params):
	return (cv2.pow((args[0] / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)


def f_exp(args, const_params):
	return (cv2.exp((args[0] / 255.0).astype(np.float32)) * 255).astype(np.uint8)

def f_log(args, const_params):
	return np.log1p(args[0]).astype(np.uint8)


def f_median_blur(args, const_params):
	ksize = correct_ksize(const_params[0] if const_params else 5)
	x = args[0].copy()
	x = x.astype(np.uint8)
	return cv2.medianBlur(x, ksize)

def f_gaussian_blur(args, const_params):
	ksize = correct_ksize(const_params[0] if const_params else 5)
	return cv2.GaussianBlur(args[0], (ksize, ksize), 0)

def f_laplacian(args, const_params):
	return cv2.Laplacian(args[0], cv2.CV_64F).astype(np.uint8)


def f_sobel(args, const_params):
	ksize = correct_ksize(const_params[0])
	if const_params[1] < 128:
		return cv2.Sobel(args[0], cv2.CV_64F, 1, 0, ksize=ksize).astype(np.uint8)
	return cv2.Sobel(args[0], cv2.CV_64F, 0, 1, ksize=ksize).astype(np.uint8)


def f_robert_cross(args, const_params):
	img = (args[0] / 255.0).astype(np.float32)
	h = cv2.filter2D(img, -1, ROBERT_CROSS_H_KERNEL)
	v = cv2.filter2D(img, -1, ROBERT_CROSS_V_KERNEL)
	return (cv2.sqrt(cv2.pow(h, 2) + cv2.pow(v, 2)) * 255).astype(np.uint8)

def f_canny(args, const_params):
	x = args[0].copy()
	x = x.astype(np.uint8)
	return cv2.Canny(x, const_params[0], const_params[1])

def f_sharpen(args, const_params):
	return cv2.filter2D(args[0], -1, SHARPEN_KERNEL)

def f_gabor_filter(args, const_params):
    ksize = 11
    gabor_k = gabor_kernel(ksize, const_params[0], const_params[1])
    return cv2.filter2D(args[0], -1, gabor_k)


def f_absolute_difference(args, const_params):
	ksize = correct_ksize(const_params[0])
	image = args[0].copy()
	return image - cv2.GaussianBlur(image, (ksize, ksize), 0) + const_params[1]

def f_absolute_difference2(args, const_params):
    return 255 - cv2.absdiff(args[0], args[1])

def f_fluo_top_hat(args, const_params):
	kernel = kernel_from_parameters(const_params)
	img = cv2.morphologyEx(args[0], cv2.MORPH_TOPHAT, kernel, iterations=10)
	kur = np.mean(kurtosis(img, fisher=True))
	skew1 = np.mean(skew(img))
	if kur > 1 and skew1 > 1:
		p2, p98 = np.percentile(img, (15, 99.5), interpolation="linear")
	else:
		p2, p98 = np.percentile(img, (15, 100), interpolation="linear")
	# rescale intensity
	output_img = np.clip(img, p2, p98)
	if p98 - p2 == 0:
		return (output_img * 255).astype(np.uint8)
	output_img = (output_img - p2) / (p98 - p2) * 255
	return output_img.astype(np.uint8)

def f_relative_difference(args, const_params):
	img = args[0]
	max_img = np.max(img)
	min_img = np.min(img)

	ksize = correct_ksize(const_params[0])
	gb = cv2.GaussianBlur(img, (ksize, ksize), 0)
	gb = np.float32(gb)

	img = np.divide(img, gb + 1e-15, dtype=np.float32)
	img = cv2.normalize(img, img, max_img, min_img, cv2.NORM_MINMAX)
	return img.astype(np.uint8)


def f_erode(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.erode(args[0], kernel)

def f_dilate(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.dilate(args[0], kernel)

def f_open(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.morphologyEx(args[0], cv2.MORPH_OPEN, kernel)

def f_close(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.morphologyEx(args[0], cv2.MORPH_CLOSE, kernel)

def f_morph_gradient(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.morphologyEx(args[0], cv2.MORPH_GRADIENT, kernel)

def f_morph_top_hat(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.morphologyEx(args[0], cv2.MORPH_TOPHAT, kernel)

def f_morph_black_hat(args, const_params):
	kernel = kernel_from_parameters(const_params)
	return cv2.morphologyEx(args[0], cv2.MORPH_BLACKHAT, kernel)

def f_fill_holes(args, const_params):
    # Contours find
    image = args[0]
    method = cv2.CHAIN_APPROX_SIMPLE
    mode = cv2.RETR_EXTERNAL
    contours = cv2.findContours(image.copy(), mode, method)[0]
    
    assert (
        len(image.shape) == 3 or len(image.shape) == 2
    ), "given image wrong format, shape must be (h, w, c) or (h, w)"
    if len(image.shape) == 3 and image.shape[-1] == 3:
        color = IMAGE_UINT8_COLOR_3C
    elif len(image.shape) == 2:
        color = IMAGE_UINT8_COLOR_1C
    else:
        raise ValueError("Image wrong format, must have 1 or 3 channels")
    selected = -1  # selects all the contours (-1)
    thickness = -1  # fills the contours (-1)
    return cv2.drawContours(image.copy(), contours, selected, color, thickness)

def f_remove_small_objects(args, const_params):
	result = remove_small_objects(args[0] > 0, const_params[0] if const_params else 64)
	return (result * 255).astype(np.uint8)


def f_remove_small_holes(args, const_params):
	result = remove_small_holes(args[0] > 0, const_params[0] if const_params else 64)
	return (result * 255).astype(np.uint8)

def f_threshold(args, const_params):
        if const_params[0] < 128:
            return cv2.threshold(args[0], const_params[1], IMAGE_UINT8_POSITIVE,  cv2.THRESH_BINARY)[1]
        return cv2.threshold(args[0], const_params[1], IMAGE_UINT8_POSITIVE, cv2.THRESH_TOZERO)[1]

def f_threshold_at_1(args, const_params):
    if const_params[0] < 128:
        return cv2.threshold(args[0], const_params[1], IMAGE_UINT8_POSITIVE,  cv2.THRESH_BINARY)[1]
    return cv2.threshold(args[0], const_params[1], IMAGE_UINT8_POSITIVE, cv2.THRESH_TOZERO)[1]


def f_distance_transform(args, const_params):
	x = args[0].copy()
	x = x.astype(np.uint8)
	return cv2.normalize(
		cv2.distanceTransform(x, cv2.DIST_L2, 3),
		None,
		0,
		255,
		cv2.NORM_MINMAX,
		cv2.CV_8U,
	)

def f_distance_transform_and_thresh(args, const_params):
	x = args[0].copy()
	x = x.astype(np.uint8)

	d = cv2.normalize(
		cv2.distanceTransform(x, cv2.DIST_L2, 3),
		None,
		0,
		255,
		cv2.NORM_MINMAX,
		cv2.CV_8U,
	)
	return cv2.threshold(d, const_params[0], IMAGE_UINT8_POSITIVE,  cv2.THRESH_BINARY)[1]


def f_binary_in_range(args, const_params):
	lower = int(min(const_params[0], const_params[1]))
	upper = int(max(const_params[0], const_params[1]))
	return cv2.inRange(args[0], lower, upper)


def f_in_range(args, const_params):
	lower = int(min(const_params[0], const_params[1]))
	upper = int(max(const_params[0], const_params[1]))
	return cv2.bitwise_and(
		args[0],
		args[0],
		mask=cv2.inRange(args[0], lower, upper),
        )


def f_min(args, const_params):
    return cv2.min(args[0], args[1])


def f_max(args, const_params):
    return cv2.max(args[0], args[1])

def f_pyr(args, const_params):
    if const_params[0] < 128:
        h, w = args[0].shape
        scaled_half = cv2.pyrDown(args[0])
        return cv2.resize(scaled_half, (w, h))
    else:
        h, w = args[0].shape
        scaled_twice = cv2.pyrUp(args[0])
        return cv2.resize(scaled_twice, (w, h))

def f_kirsch(args, const_params):
    x = args[0].copy()
    x = x.astype(np.uint8)
    compass_gradients = [
        cv2.filter2D(x, ddepth=cv2.CV_32F, kernel=kernel/5)
        for kernel in KERNEL_KIRSCH_COMPASS
    ]
    res = np.max(compass_gradients, axis=0)
    res[res > 255] = 255
    return res.astype(np.uint8)

def f_embossing(args, const_params):
    x = args[0].copy()
    x = x.astype(np.uint8)
    res = cv2.filter2D(x, ddepth=cv2.CV_32F, kernel=KERNEL_EMBOSS)
    res[res > 255] = 255
    return res.astype(np.uint8)

def f_normalization(args, const_params):
    return cv2.normalize(args[0],  None, 0, 255, cv2.NORM_MINMAX)

def f_denoizing(args, const_params):
    x = args[0].copy()
    x = x.astype(np.uint8)
    return cv2.fastNlMeansDenoising(x, None, h=np.uint8(const_params[0]))

def f_threshold_otsu(args, const_params):
    x = args[0].copy()
    x = x.astype(np.uint8)
    return cv2.threshold(x, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)[1]
 
def kernel_from_parameters(p):
    # 50%
    if p[1] < 128:
        return ellipse_kernel(p[0])
    # 25%
    if p[1] < 192:
        return cross_kernel(p[0])
    # 25%
    return rect_kernel(p[0])

KERNEL_EMBOSS = np.array(([-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]), dtype="int")

KERNEL_KIRSCH_N = np.array(([5, 5, 5],
                          [-3, 0, -3],
                          [-3, -3, -3]), dtype="int")

KERNEL_KIRSCH_NE = np.array(([-3, 5, 5],
                          [-3, 0, 5],
                          [-3, -3, -3]), dtype="int")

KERNEL_KIRSCH_E = np.array(([-3, -3, 5],
                          [-3, 0, 5],
                          [-3, -3, 5]), dtype="int")

KERNEL_KIRSCH_SE = np.array(([-3, -3, -3],
                          [-3, 0, 5],
                          [-3, 5, 5]), dtype="int")

KERNEL_KIRSCH_S = np.array(([-3, -3, -3],
                          [-3, 0, -3],
                          [5, 5, 5]), dtype="int")

KERNEL_KIRSCH_SW = np.array(([-3, -3, -3],
                          [5, 0, -3],
                          [5, 5, -3]), dtype="int")

KERNEL_KIRSCH_W = np.array(([5, -3, -3],
                          [5, 0, -3],
                          [5, -3, -3]), dtype="int")

KERNEL_KIRSCH_NW = np.array(([5, 5, -3],
                          [5, 0, -3],
                          [-3, -3, -3]), dtype="int")

KERNEL_KIRSCH_COMPASS = [
    KERNEL_KIRSCH_N,
    KERNEL_KIRSCH_NE,
    KERNEL_KIRSCH_E,
    KERNEL_KIRSCH_SE,
    KERNEL_KIRSCH_S,
    KERNEL_KIRSCH_SW,
    KERNEL_KIRSCH_W,
    KERNEL_KIRSCH_NW]


def ellipse_kernel(ksize):
    ksize = correct_ksize(ksize)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))


def cross_kernel(ksize):
    ksize = correct_ksize(ksize)
    return cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))


def rect_kernel(ksize):
    ksize = correct_ksize(ksize)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))


def gabor_kernel(ksize, p1, p2):
    ksize = clamp_ksize(ksize)
    ksize = unodd_ksize(ksize)
    p1_bin = "{0:08b}".format(p1)
    p2_bin = "{0:08b}".format(p2)

    sigma = GABOR_SIGMAS[int(p1_bin[:4], 2)]
    theta = GABOR_THETAS[int(p1_bin[4:], 2)]
    lambd = GABOR_LAMBDS[int(p2_bin[:4], 2)]
    gamma = GABOR_GAMMAS[int(p2_bin[4:], 2)]

    return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)

def clamp_ksize(ksize):
    if ksize < OPENCV_MIN_KERNEL_SIZE:
        return OPENCV_MIN_KERNEL_SIZE
    if ksize > OPENCV_MAX_KERNEL_SIZE:
        return OPENCV_MAX_KERNEL_SIZE
    return ksize


def remap_ksize(ksize):
    return int(round(ksize * KERNEL_SCALE + OPENCV_MIN_KERNEL_SIZE))


def unodd_ksize(ksize):
    if ksize % 2 == 0:
        return ksize + 1
    return ksize


def correct_ksize(ksize):
    ksize = remap_ksize(ksize)
    ksize = clamp_ksize(ksize)
    ksize = unodd_ksize(ksize)
    return ksize
