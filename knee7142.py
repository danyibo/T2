# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os, math, cv2
import numpy as np
import nibabel as nib
from glob import glob
from skimage import data, io, filters
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage.morphology import binary_dilation, binary_closing, binary_opening, grey_closing
from scipy.ndimage import generate_binary_structure
from skimage.filters import threshold_minimum, threshold_otsu, threshold_local
#from MeDIT.Visualization import Imshow3DArray

# %%
# 8-neighbor connectivity
def get_neighbors_8(pos):
    return [ (pos[0]+1, pos[1]),  (pos[0]-1, pos[1]),  (pos[0], pos[1]+1),  (pos[0], pos[1]-1),
             (pos[0]+1, pos[1]+1),(pos[0]-1, pos[1]+1),(pos[0]+1,pos[1]-1), (pos[0]-1, pos[1]-1) ]

# 4-neighbor connectivity
def get_neighbors_4(pos):
    return [ (pos[0]+1, pos[1]),  (pos[0]-1, pos[1]),  (pos[0], pos[1]+1),  (pos[0], pos[1]-1)]

# if zero px neighbor exists, return true
def zero_neighbors(image, pos):
    # if image[pos[0]+1, pos[1]]==0 or image[pos[0]-1, pos[1]]==0 or \
    #    image[pos[0], pos[1]+1]==0 or image[pos[0], pos[1]-1]==0:
    if np.any([image[pt[0], pt[1]]==0 for pt in get_neighbors_4(pos)]):
        return True
    else:
        return False

def end_points(line, pos):
    n = int(line[pos[0]+1, pos[1]]>0) + \
        int(line[pos[0]-1, pos[1]]>0) + \
        int(line[pos[0], pos[1]+1]>0) + \
        int(line[pos[0], pos[1]-1]>0) + \
        int(line[pos[0]+1, pos[1]-1]>0) + \
        int(line[pos[0]-1, pos[1]-1]>0) + \
        int(line[pos[0]+1, pos[1]+1]>0) + \
        int(line[pos[0]-1, pos[1]+1]>0) 

    if n == 1:
        return True
    else:
        return False

def boundingbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def z_range(binary_image, axis=(0,1)):
    "return non-zero z axis slices (min, max+1)"
    z = np.any(binary_image, axis=axis)     # 只要有一个为真，则返回true
    zmin, zmax = np.where(z)[0][[0, -1]]
    return zmin, zmax+1

def z_nonzero_slices(binary_image, axis=(0,1)):
    z = np.any(binary_image, axis=axis)
    slices = np.where(z)[0]
    return slices

def pt_in_array(pt, points, axis=1):
    if isinstance(points, (np.ndarray, np.generic)):
        found = np.where((points==pt).all(axis=axis))[0]
        return len(found) > 0
    elif isinstance(points, list):
        return pt in points

def recheck_endpoint(boundary, end_point):
    bound = boundary.squeeze()
    if pt_in_array(end_point, bound):
        return end_point  # end_point is in boundary
    else:
        for neighbor in get_neighbors_8(end_point):
            if pt_in_array(neighbor, bound):
                return neighbor
        raise NotImplementedError(f'New end point not found! Whats next!?\n {bound}, {end_point}')

def get_boundary_2d(binary_slice):
    coord = list(zip(*np.where(binary_slice)))
    bound = list(filter(lambda x: zero_neighbors(binary_slice, x), coord))
    return bound

def get_boundary_3d(binary_mask, region_label):
    boundaries = []
    slices = np.arange(*z_range(binary_mask))
    for idx in slices:
        bin_slice = binary_mask[..., idx]
        boundary = get_boundary_2d(bin_slice==region_label)
        boundaries.append(boundary)
    return boundaries, slices

def get_next_pt(prev_pt, all_pts):
    next_pts = list(filter(lambda x : x in all_pts, get_neighbors_8(prev_pt)))
    assert len(next_pts) > 0, 'No next pt is found!'
    return next_pts[0]

def __split_boundary(boundary, endpts):
    #assert len(endpts) == 2, f'Incorrect end ponits found! len: {len(endpts)} != 2'
    new_endpts = [ recheck_endpoint(boundary, pt) for pt in endpts ]

    segments = []
    segment  = []
    processed_pts = []
    unprocessed_pts = boundary.copy()
    for endpt in new_endpts:
        if endpt in processed_pts:
            continue
        processed_pts.append(endpt)
        unprocessed_pts.remove(endpt)
        segment.append(endpt)

        while True:
            next_pt = get_next_pt(processed_pts[-1], unprocessed_pts)
            if next_pt in new_endpts:
                segments.append(segment)
                break
            else:
                segment.append(next_pt)
                processed_pts.append(next_pt)
                unprocessed_pts.remove(next_pt)
    return segments

def split_boundary_cv(boundaries, endpts):
    segments = []
    for bound in boundaries:
        # print(f'{len(bound)} points are found!! Shape: {bound.shape}')
        bound = bound.squeeze()
        #curr_endpt = list(filter(lambda x : x in bound, endpts))
        #curr_endpt_idx = [ bound.index(pt) for pt in curr_endpt ]
        curr_endpt_idx = []
        for pt in endpts:
            found = np.where((bound==pt).all(axis=1))[0]
            if len(found) > 0:
                curr_endpt_idx.append(found[0])
        assert len(curr_endpt_idx) == 2, f'Not found two end ponts: len {len(curr_endpt_idx)}'

        segment_1 = bound[min(curr_endpt_idx):max(curr_endpt_idx),:]
        segment_2 = np.array(list(filter( lambda x: not pt_in_array(x, segment_1), bound )))
        segments.append(segment_1)
        segments.append(segment_2)
    return segments

def get_boundary_cv(binary_slice):
    # because opencv can return ordered boundary points
    slice_cv = cv2.cvtColor(binary_slice.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(slice_cv, cv2.COLOR_BGR2GRAY)
    retval, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contour, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contour

# judge whether line segment has intersection with target region
def intersect(p1, p2, region, nb_points=40, dilation=0):
    """"Return if a line segment between points p1 and p2
    has intersection with region"""
    # If we have 8 intermediate points, we have 8+1=9 spaces between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    if dilation>0:
        dil_region = binary_dilation(region, generate_binary_structure(2,1), dilation)
    else:
        dil_region = region
    ret = np.any([dil_region[int(p1[0]+i*x_spacing), int(p1[1]+i*y_spacing)]
           for i in range(1, nb_points+1)])
    return ret

def intersect2(p1, p2, line, nb_points=50):
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    dil_points = []
    for i in range(4, nb_points+1):
        dil_points += get_neighbors_8((int(p1[0]+i*x_spacing), int(p1[1]+i*y_spacing)))

    return np.any([pt_in_array(tuple(pt), dil_points) for pt in line])

def get_line_segment(p1, p2, nb_points=30):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [(int(p1[0]+i*x_spacing), int(p1[1]+i*y_spacing))
           for i in range(1, nb_points+1)]

def get_delta_xy(pt, reso, poly_fn, delta_dist, k=None):
    '''
    Get x&y px-wise delta of given pt from specified curve fn
    if k is given, directly use k, ignore poly_fn
    '''
    x = pt[0]
    if k is None:
        k = -1 / poly_fn.deriv(1)(x) # norm direction

    delta_x = delta_dist / math.sqrt(1+k*k)
    delta_y = delta_dist / math.sqrt(1+k*k) * k
    delta_x_px = int(delta_x / reso[0])
    delta_y_px = int(delta_y / reso[1])
    return delta_x_px, delta_y_px

def get_direction(coords, ):
    pos_direction = False
    center_idx = len(coords[0])//2
    center_pt = (coords[0][center_idx], coords[1][center_idx])
    center_delta = get_delta_xy(center_pt, reso, p, 10)
    center_pt_pos = (center_pt[0]+center_delta[0], center_pt[1]+center_delta[1])
    if intersect(center_pt, center_pt_pos, region_2[...,slice_idx]):
        pos_direction = True
    print('Positive direction:', pos_direction)

def is_acute_angle(pt1, pt2, direction):
    new_direction = (pt2[0]-pt1[0], pt2[1]-pt1[1])
    if new_direction[0]*direction[0] + new_direction[1]*direction[1] > 0:
        return True
    else:
        return False

def norm_process(img_path, roi_path, out_label):
    image_nii = nib.load(img_path)
    label_nii = nib.load(roi_path)
    image = np.copy(image_nii.get_data())
    label = np.copy(label_nii.get_data())
    reso = image_nii.header['pixdim'][1:4]
    assert image.shape == label.shape, 'Image shape != Label shape'
    print('Image size:', image.shape, 'image reso:', reso)

    # Get threshold for boundary
    thresh_min = threshold_minimum(image[label>0])
    thresh_otsu = threshold_otsu(image[label>0])

    # ax = plt.hist(image[label>0].ravel(), bins = 64)
    # plt.axvline(thresh_min, color='r')
    # plt.axvline(thresh_otsu, color='g')
    # plt.show()

    region_1 = np.logical_and(image>=thresh_otsu, label>0).astype(np.int8)
    region_2 = np.logical_and(image<thresh_otsu, label>0).astype(np.int8)

    new_label = np.zeros_like(label)
    new_label[region_1>0] = 2
    new_label[region_2>0] = 3

    # slice-wise
    boundaries, slice_indices = get_boundary_3d(new_label, 2)
    print(f'Found valid {len(slice_indices)} slices:', slice_indices)

    margin_label = np.zeros_like(new_label)
    for bound, slice_idx in zip(boundaries, slice_indices):
        new_slice = np.zeros(region_1.shape[:2]).astype(np.int)
        for pt in bound:
            new_slice[pt[0], pt[1]] = 1

        #plt.imshow(new_slice, vmin=0, vmax=1)

        order = 3
        delta_dist = 5 # 5mm
        coords = np.array(bound).transpose()
        z = np.polyfit(coords[0], coords[1], order)
        p = np.poly1d(z)

        # # judge direction
        # pos_direction = False
        # center_idx = len(coords[0])//2
        # center_pt = (coords[0][center_idx], coords[1][center_idx])
        # center_delta = get_delta_xy(center_pt, reso, p, delta_dist)
        # center_pt_pos = (center_pt[0]+center_delta[0], center_pt[1]+center_delta[1])
        # if intersect(center_pt, center_pt_pos, region_2[...,slice_idx]):
        #     pos_direction = True
        # print('Positive direction:', pos_direction)
        
        new_line = []
        for x, y in zip(coords[0], coords[1]):
            delta_x_px, delta_y_px = get_delta_xy((x,y), reso, p, delta_dist)
            new_pt_pos = (x+delta_x_px, y+delta_y_px)
            new_pt_neg = (x-delta_x_px, y-delta_y_px)
            if intersect((x,y), new_pt_pos, region_2[...,slice_idx]):
                inter_pts = get_line_segment((x,y), new_pt_pos)
                new_line = new_line + inter_pts
            else:
                inter_pts = get_line_segment((x,y), new_pt_neg)
                new_line = new_line + inter_pts

        for pt in new_line:
            new_slice[pt[0], pt[1]] = out_label+8
        new_slice = binary_closing(new_slice, generate_binary_structure(2,1), iterations=2).astype(np.int)
        new_slice = binary_opening(new_slice,generate_binary_structure(2,1),iterations=1).astype(np.int)
        new_slice[new_slice>0] = out_label+8
        margin_label[..., slice_idx] = new_slice

    margin_label[new_label==2] = out_label
    #return margin_label
    nib.save(nib.Nifti1Image(margin_label, image_nii.affine, image_nii.header), 
             os.path.join(out_dir, os.path.basename(roi_path)))
    #nib.save(nib.Nifti1Image(new_label, image_nii.affine, image_nii.header), 
    #         os.path.join(out_dir, os.path.basename(roi_path).replace('.nii', '_new.nii')))

def shift_process(img_path, roi_path, out_label):
    image_nii = nib.load(img_path)
    label_nii = nib.load(roi_path)
    image = np.copy(image_nii.get_data())
    label = np.copy(label_nii.get_data())
    reso = image_nii.header['pixdim'][1:4]
    assert image.shape == label.shape, 'Image shape != Label shape'
    print('Image size:', image.shape, 'image reso:', reso)

    # Get threshold for boundary
    thresh_min = threshold_minimum(image[label>0])
    thresh_otsu = threshold_otsu(image[label>0])

    region_1 = np.logical_and(image>thresh_otsu, label>0).astype(np.int8)
    region_2 = np.logical_and(image<=thresh_otsu, label>0).astype(np.int8)

    new_label = np.zeros_like(label)
    new_label[region_1>0] = 2
    new_label[region_2>0] = 3

    # slice-wise
    boundaries, slice_indices = get_boundary_3d(new_label, 2)
    print(f'Found valid {len(slice_indices)} slices:', slice_indices)

    margin_label = np.zeros_like(new_label)
    for bound, slice_idx in zip(boundaries, slice_indices):
        new_slice = np.zeros(region_1.shape[:2]).astype(np.int)
        for pt in bound:
            new_slice[pt[0], pt[1]] = 1

        order = 5
        delta_dist = 5 # 5mm
        coords = np.array(bound).transpose()
        z = np.polyfit(coords[0], coords[1], order)
        p = np.poly1d(z)

        # judge direction
        pos_direction = False
        center_idx = len(coords[0])//2
        center_pt = (coords[0][center_idx], coords[1][center_idx])
        center_delta = get_delta_xy(center_pt, reso, p, 10)
        center_pt_pos = (center_pt[0]+center_delta[0], center_pt[1]+center_delta[1])
        if intersect(center_pt, center_pt_pos, region_2[...,slice_idx]):
            pos_direction = True
        print('Positive direction:', pos_direction)
        
        new_line = []
        for x, y in zip(coords[0], coords[1]):
            delta_x_px, delta_y_px = get_delta_xy((x,y), reso, p, delta_dist, k=-1/p.deriv(1)(center_pt[0]))
            new_pt_pos = (x+delta_x_px, y+delta_y_px)
            new_pt_neg = (x-delta_x_px, y-delta_y_px)
            if pos_direction:
                inter_pts = get_line_segment((x,y), new_pt_pos)
                new_line = new_line + inter_pts
            else:
                inter_pts = get_line_segment((x,y), new_pt_neg)
                new_line = new_line + inter_pts

        for pt in new_line:
            new_slice[pt[0], pt[1]] = out_label+8
        new_slice = binary_closing(new_slice, generate_binary_structure(2,1), iterations=1).astype(np.int)
        new_slice[new_slice>0] = out_label+8
        margin_label[..., slice_idx] = new_slice

    margin_label[new_label==2] = out_label
    #return margin_label
    nib.save(nib.Nifti1Image(margin_label, image_nii.affine, image_nii.header), 
             os.path.join(out_dir, os.path.basename(roi_path)))

def plot_lines(line1, points, line2=None, color='cyan'):
    #plt.imshow(binary_slice)
    plt.plot(np.array(line1)[:, 1], np.array(line1)[:, 0], color=color, marker='o',
            linestyle='None', markersize=1)
    plt.plot(np.array(points)[:, 1], np.array(points)[:, 0], color='red', marker='o',
            linestyle='None', markersize=6)
    if line2 is not None:
        plt.plot(np.array(line2)[:, 1], np.array(line2)[:, 0], color='green', marker='o',
            linestyle='None', markersize=1)
    plt.xlim(50, 300)
    plt.ylim(50, 300)
    # plt.show()

def get_gravity(binary_slice):
    coords = list(zip(*np.where(binary_slice>0)))
    x, y = 0, 0
    for pt in coords:
        x += pt[0]
        y += pt[1]
    return x//len(coords), y//len(coords)

def classify_boundary(line1, line2, binary_slice, target_label, slice_ref):
    if target_label in [1,2]: # tmp fix!
        origin_pt = get_gravity(binary_slice)        
        dist1 = min([(p[0]-origin_pt[0])**2+(p[1]-origin_pt[1])**2 for p in line1])
        dist2 = min([(p[0]-origin_pt[0])**2+(p[1]-origin_pt[1])**2 for p in line2])
        return dist1 < dist2
    else:
        tmp_coords = list(zip(*np.where(slice_ref>0)))
        xmin, xmax, ymin, ymax = boundingbox(binary_slice)
        tmp_coords = list(filter(lambda x: xmin-50<x[0]<xmax+50 and ymin-50<x[1]<ymax+50, tmp_coords))
        # origin_pt = tmp_coords[np.random.randint(len(tmp_coords))]

        dist1 = np.mean([p[1] for p in line1])
        dist2 = np.mean([p[1] for p in line2])

        return dist1 > dist2

        # nb_points = 30
        # x_spacing = (pt1[0] - origin_pt[0]) / (nb_points + 1)
        # y_spacing = (pt1[1] - origin_pt[1]) / (nb_points + 1)

        # dist1 = dist2 = -1
        # for i in range(1, nb_points*2):
        #     dest = get_neighbors_4((int(origin_pt[0]+i*x_spacing), int(origin_pt[1]+i*y_spacing)))
        #     #dest = (int(origin_pt[0]+i*x_spacing), int(origin_pt[1]+i*y_spacing))
        #     if np.any([pt_in_array(tuple(p), dest) for p in line1]):
        #         dist1 = i
        #     if np.any([pt_in_array(tuple(p),dest) for p in line2]):
        #         dist2 = i
        #     if dist1 > 0 and dist2 > 0:
        #         break
        # return dist1 > dist2

#%%
'''
if __name__ == '__main__bak__':
    # Read data and mask
    root_dir = r"E:\doctor tao\data\Ankle instability\DICOMDJW"
    out_dir  = os.path.join(root_dir, 'result')
    os.makedirs(out_dir, exist_ok=True)
    img_regx = 'src.nii'
    nii_paths = glob(os.path.join(root_dir, '*.nii'))

    img_paths = list(filter(lambda x: 'src.nii' in x, nii_paths))
    roi_paths = list(filter(lambda x: 'src.nii' not in x, nii_paths))
    assert len(img_paths)==1, 'No or multiple src image found!'
    assert len(roi_paths)>0, 'ROI not exists!'
    img_path = img_paths[0]

    CATEGORIES = {
        'L subtalar cal':1, 'L subtalar talus':2,
        'L talus dome':3, 'L tibial':4,
        'M subtalar cal':5, 'M subtalar talus':6,
        'M talus dome':7, 'M tibial':8
    }
    STRATEGY = 'norm' #'norm'
    # shift
    # STRATEGY = 'shift'

    image_nii = nib.load(img_path)

    prefix = os.path.basename(img_path).replace(img_regx,'')
    #final_mask = np.zeros(image_nii.shape)
    for roi_path in roi_paths:
        label_idx = CATEGORIES[os.path.basename(roi_path).replace(prefix,'').strip('.nii')]
        print('Processing label:', label_idx)

        if STRATEGY == 'norm':
            mask = norm_process(img_path, roi_path, label_idx)
            #final_mask = final_mask+mask
        elif STRATEGY == 'shift':
            mask = shift_process(img_path, roi_path, label_idx)

    # nib.save(nib.Nifti1Image(final_mask, image_nii.affine, image_nii.header), 
    #          os.path.join(out_dir, 'results.nii'))'''



# %% Padding program for knee 
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
def padd(roi_fname):

    # roi_fname = r"E:\Data\doctor_xie\knee_data\CAI NAI JI-20140605\roi.nii.gz"
    nii = nib.load(roi_fname)
    roi = nii.get_fdata() # 必须使用这种方法
    reso = nii.header['pixdim'][1:4]

    # print('shape:', roi.shape, 'labels:', np.unique(roi))
    ref_regions = {1:5, 3:5, 2:6, 4:6}

    output_roi = np.zeros_like(roi)
    for slice_idx in z_nonzero_slices(roi):
        img_slice = roi[...,slice_idx]
        output_roi_slice = np.zeros_like(img_slice)
        #plt.imshow(img_slice)
        #plt.show()
        unique_labels = np.unique(img_slice)
        candidate_label = np.intersect1d(unique_labels, [1,2,3,4])
        for target_label in candidate_label:
            # print('Processing label:', target_label)
            binary_slices = []
            slice_image = np.array(img_slice==target_label).astype(np.int8)

            #labeling. for handling multiple components
            labeled_array, num_features = label(slice_image, structure=generate_binary_structure(2,3))
            bins = np.bincount(labeled_array.ravel())
            component_idx = np.where(bins[1:]>60)[0]
            for idx in component_idx:
                binary_slices.append(labeled_array==idx+1)

            for binary_slice in binary_slices:
                skeleton = skeletonize(binary_slice).astype(np.int8)
                coord = list(zip(*np.where(skeleton>0)))
                endpts = list(filter(lambda x: end_points(skeleton, x), coord))

                # get the outer boundary
                boundary = get_boundary_cv(binary_slice.transpose()>0) #opencv image rotated 90
                # split the boundary
                new_endpts = [ recheck_endpoint(boundary[0], pt) for pt in endpts ]
                # print('old endpoints:', endpts, 'new endpoints:', new_endpts)
                boundaries = split_boundary_cv(boundary, new_endpts)
                #plot_lines(boundary[0].squeeze(), endpts)

                # judge which boundary is our target!
                ref_label = ref_regions[target_label]
                slice_ref = np.array(img_slice==ref_label).astype(np.int8)
                if classify_boundary(boundaries[0], boundaries[1], binary_slice, target_label, slice_ref):
                    target_boundary = boundaries[0]
                    ref_boundary = boundaries[1]
                else:
                    target_boundary = boundaries[1]
                    ref_boundary = boundaries[0]

                # Curve fit
                order = 3
                delta_dist = 5 # 5mm
                coords = np.array(target_boundary).transpose()
                z = np.polyfit(coords[0], coords[1], order)
                p = np.poly1d(z)


                center_idx = len(coords[0])//2
                center_pt = (coords[0][center_idx], coords[1][center_idx])
                center_delta = get_delta_xy(center_pt, reso, p, 10)
                center_pt_pos = (center_pt[0]+center_delta[0], center_pt[1]+center_delta[1])
                if intersect2(center_pt, center_pt_pos, ref_boundary):
                    direction = (-center_delta[0], -center_delta[1])
                else:
                    direction = (center_delta[0], center_delta[1])
                # print('Padding direction:', direction)

                # generate continous curve
                # x_min, x_max, num_points = np.min(coords[0]), np.max(coords[0]), coords.shape[1]*2
                # continous_curve = [ [x, p(x)] for x in np.linspace(x_min, x_max, num=num_points) ]
                # print('x_min, x_max, num_points:', x_min, x_max, num_points)
                # plot_lines(continous_curve, endpts, color='blue')
                # plot_lines(coords.transpose(), endpts, color='green')
                # continue

                # Compute margin direction
                def get_padding_region(x, y, direction, ref_boundary, reso, p, delta_dist):
                    new_line = []
                    delta_x_px, delta_y_px = get_delta_xy((x,y), reso, p, delta_dist)
                    new_pt_pos = (x+delta_x_px, y+delta_y_px)
                    new_pt_neg = (x-delta_x_px, y-delta_y_px)
                    if is_acute_angle((x,y), new_pt_pos, direction):
                        inter_pts = get_line_segment((x,y), new_pt_pos)
                        if np.any([ pt_in_array(pt, ref_boundary) for pt in inter_pts]):
                            inter_pts =  get_line_segment((x,y), new_pt_neg)
                        new_line = new_line + inter_pts
                        return new_line #, (delta_x_px, delta_y_px)
                    elif is_acute_angle((x,y), new_pt_neg, direction):
                        inter_pts = get_line_segment((x,y), new_pt_neg)
                        if np.any([pt_in_array(pt, ref_boundary) for pt in inter_pts]):
                            inter_pts =  get_line_segment((x,y), new_pt_pos)
                        new_line = new_line + inter_pts
                        return new_line# , (-delta_x_px, -delta_y_px)
                    else:
                        # print('no intersection')
                        return []

                new_line = []
                for x, y in zip(coords[0], coords[1]):
                    padding_lines = get_padding_region(x, y, direction, ref_boundary, reso, p, delta_dist)
                    new_line += padding_lines

                plot_lines(new_line, endpts, color='green')
                tmp_slice = np.zeros_like(output_roi_slice)
                for pt in new_line:
                    tmp_slice[pt[0], pt[1]] = target_label
                for pt in target_boundary:
                    tmp_slice[pt[0], pt[1]] = 10
                tmp_slice = binary_closing(tmp_slice, generate_binary_structure(2,2), iterations=1).astype(np.int)
                output_roi_slice[tmp_slice > 0] = target_label
        output_roi[...,slice_idx] = output_roi_slice

    nib.save(nib.Nifti1Image(output_roi, nii.affine, nii.header),
             os.path.join(roi_fname.replace('.nii', '_new.nii')))

# padd(roi_fname=r"Y:\DYB\2020832DATA\doctor_xie\Normal_data_nii_and_T2\case0\roi.nii.gz")
all_data_path =r"Y:\DYB\2020832DATA\doctor_xie\normal_control"
for case in os.listdir(all_data_path):
    roi_path = os.path.join(all_data_path, case, "new_roi.nii")
    padd(roi_fname=roi_path)
    # try:
    #     padd(roi_fname=roi_path)
    # except:
    #     print(case)