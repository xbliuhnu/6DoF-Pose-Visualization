# -*- coding: utf-8 -*-
"""
@file   : visualize.py
@author : liuxb
@email  : liuxuebing@hnu.edu.cn
@product: PyCharm
Created on 2021/3/17 下午10:53
"""

import os
import numpy as np
import cv2


# this function forked from 'https://github.com/cvlab-epfl/segmentation-driven-pose/blob/master/utils.py'
def get_class_colors(class_id):
    colordict = {'gray': [128, 128, 128], 'silver': [192, 192, 192], 'black': [0, 0, 0],
                 'maroon': [128, 0, 0], 'red': [255, 0, 0], 'purple': [128, 0, 128], 'fuchsia': [255, 0, 255],
                 'green': [0, 128, 0],
                 'lime': [0, 255, 0], 'olive': [128, 128, 0], 'yellow': [255, 255, 0], 'navy': [0, 0, 128],
                 'blue': [0, 0, 255],
                 'teal': [0, 128, 128], 'aqua': [0, 255, 255], 'orange': [255, 165, 0], 'indianred': [205, 92, 92],
                 'lightcoral': [240, 128, 128], 'salmon': [250, 128, 114], 'darksalmon': [233, 150, 122],
                 'lightsalmon': [255, 160, 122], 'crimson': [220, 20, 60], 'firebrick': [178, 34, 34],
                 'darkred': [139, 0, 0],
                 'pink': [255, 192, 203], 'lightpink': [255, 182, 193], 'hotpink': [255, 105, 180],
                 'deeppink': [255, 20, 147],
                 'mediumvioletred': [199, 21, 133], 'palevioletred': [219, 112, 147], 'coral': [255, 127, 80],
                 'tomato': [255, 99, 71], 'orangered': [255, 69, 0], 'darkorange': [255, 140, 0], 'gold': [255, 215, 0],
                 'lightyellow': [255, 255, 224], 'lemonchiffon': [255, 250, 205],
                 'lightgoldenrodyellow': [250, 250, 210],
                 'papayawhip': [255, 239, 213], 'moccasin': [255, 228, 181], 'peachpuff': [255, 218, 185],
                 'palegoldenrod': [238, 232, 170], 'khaki': [240, 230, 140], 'darkkhaki': [189, 183, 107],
                 'lavender': [230, 230, 250], 'thistle': [216, 191, 216], 'plum': [221, 160, 221],
                 'violet': [238, 130, 238],
                 'orchid': [218, 112, 214], 'magenta': [255, 0, 255], 'mediumorchid': [186, 85, 211],
                 'mediumpurple': [147, 112, 219], 'blueviolet': [138, 43, 226], 'darkviolet': [148, 0, 211],
                 'darkorchid': [153, 50, 204], 'darkmagenta': [139, 0, 139], 'indigo': [75, 0, 130],
                 'slateblue': [106, 90, 205],
                 'darkslateblue': [72, 61, 139], 'mediumslateblue': [123, 104, 238], 'greenyellow': [173, 255, 47],
                 'chartreuse': [127, 255, 0], 'lawngreen': [124, 252, 0], 'limegreen': [50, 205, 50],
                 'palegreen': [152, 251, 152],
                 'lightgreen': [144, 238, 144], 'mediumspringgreen': [0, 250, 154], 'springgreen': [0, 255, 127],
                 'mediumseagreen': [60, 179, 113], 'seagreen': [46, 139, 87], 'forestgreen': [34, 139, 34],
                 'darkgreen': [0, 100, 0], 'yellowgreen': [154, 205, 50], 'olivedrab': [107, 142, 35],
                 'darkolivegreen': [85, 107, 47], 'mediumaquamarine': [102, 205, 170], 'darkseagreen': [143, 188, 143],
                 'lightseagreen': [32, 178, 170], 'darkcyan': [0, 139, 139], 'cyan': [0, 255, 255],
                 'lightcyan': [224, 255, 255],
                 'paleturquoise': [175, 238, 238], 'aquamarine': [127, 255, 212], 'turquoise': [64, 224, 208],
                 'mediumturquoise': [72, 209, 204], 'darkturquoise': [0, 206, 209], 'cadetblue': [95, 158, 160],
                 'steelblue': [70, 130, 180], 'lightsteelblue': [176, 196, 222], 'powderblue': [176, 224, 230],
                 'lightblue': [173, 216, 230], 'skyblue': [135, 206, 235], 'lightskyblue': [135, 206, 250],
                 'deepskyblue': [0, 191, 255], 'dodgerblue': [30, 144, 255], 'cornflowerblue': [100, 149, 237],
                 'royalblue': [65, 105, 225], 'mediumblue': [0, 0, 205], 'darkblue': [0, 0, 139],
                 'midnightblue': [25, 25, 112],
                 'cornsilk': [255, 248, 220], 'blanchedalmond': [255, 235, 205], 'bisque': [255, 228, 196],
                 'navajowhite': [255, 222, 173], 'wheat': [245, 222, 179], 'burlywood': [222, 184, 135],
                 'tan': [210, 180, 140],
                 'rosybrown': [188, 143, 143], 'sandybrown': [244, 164, 96], 'goldenrod': [218, 165, 32],
                 'darkgoldenrod': [184, 134, 11], 'peru': [205, 133, 63], 'chocolate': [210, 105, 30],
                 'saddlebrown': [139, 69, 19],
                 'sienna': [160, 82, 45], 'brown': [165, 42, 42], 'snow': [255, 250, 250], 'honeydew': [240, 255, 240],
                 'mintcream': [245, 255, 250], 'azure': [240, 255, 255], 'aliceblue': [240, 248, 255],
                 'ghostwhite': [248, 248, 255], 'whitesmoke': [245, 245, 245], 'seashell': [255, 245, 238],
                 'beige': [245, 245, 220], 'oldlace': [253, 245, 230], 'floralwhite': [255, 250, 240],
                 'ivory': [255, 255, 240],
                 'antiquewhite': [250, 235, 215], 'linen': [250, 240, 230], 'lavenderblush': [255, 240, 245],
                 'mistyrose': [255, 228, 225], 'gainsboro': [220, 220, 220], 'lightgrey': [211, 211, 211],
                 'darkgray': [169, 169, 169], 'dimgray': [105, 105, 105], 'lightslategray': [119, 136, 153],
                 'slategray': [112, 128, 144], 'darkslategray': [47, 79, 79], 'white': [255, 255, 255]}

    colornames = list(colordict.keys())
    assert (class_id < len(colornames))

    r, g, b = colordict[colornames[class_id]]

    return r, g, b


def load_pts(root_dir):
    # get model pts, only for YCB datasets

    models_dir = os.path.join(root_dir, 'models')
    models_cls = os.listdir(models_dir)
    models_cls.sort()
    models_pts = []
    for cls in models_cls:
        file = os.path.join(models_dir, cls, 'points.xyz')
        pts = []
        with open(file, 'r') as f:
            for line in f.readlines():
                chs = line.split()
                pts.append(list(map(float, chs[:3])))
        models_pts.append(pts)
    models_pts = np.array(models_pts)
    model_bk = np.expand_dims(np.zeros_like(models_pts[0]), axis=0)  # background points
    models_pts = np.concatenate((model_bk, models_pts), axis=0)

    return models_pts


def visualize_contour(image, obj_list, pts_all, pose_est, pose_gt, k):
    # image, [h, w, 3]
    # obj_list, [n], number of object in image
    # pts_all, [cls, l, 3], model pts of all object
    # pose_est, [n, 3, 4], estimation pose
    # pose_gt, [n, 3, 4], ground truth pose
    # k, dict, camera intrinsic

    h, w, _ = image.shape
    mask_img = np.zeros((h, w), np.uint8)
    kernel = np.ones((6, 6), np.uint8)
    dst_img = np.copy(image)
    pts_obj = pts_all[obj_list]         # [n, l, 3]
    pts_obj = np.concatenate((pts_obj, np.ones([pts_obj.shape[0], pts_obj.shape[1], 1], dtype=np.float)), axis=2)       # [n, l, 4]
    pts_obj = pts_obj.transpose([0, 2, 1])          # [n, 4, l]
    pts_gt = np.matmul(pose_gt, pts_obj)            # [n, 3, l]
    pts_est = np.matmul(pose_est, pts_obj)          # [n, 3, l]

    xs_gt = k['fx'] * pts_gt[:, 0, :] / pts_gt[:, 2, :] + k['cx']
    ys_gt = k['fy'] * pts_gt[:, 1, :] / pts_gt[:, 2, :] + k['cy']
    xs_gt = np.rint(xs_gt).astype(np.int)
    ys_gt = np.rint(ys_gt).astype(np.int)

    for idx in range(obj_list.shape[0]):
        mask_img.fill(0)
        xs = xs_gt[idx]
        ys = ys_gt[idx]
        for x, y in zip(xs, ys):
            cv2.circle(mask_img, (x, y), 1, 255, -1)

        # fill the holes
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
        # find contour
        contours, _ = cv2.findContours(mask_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        dst_img = cv2.drawContours(dst_img, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

    xs_est = k['fx'] * pts_est[:, 0, :] / pts_est[:, 2, :] + k['cx']
    ys_est = k['fy'] * pts_est[:, 1, :] / pts_est[:, 2, :] + k['cy']
    xs_est = np.rint(xs_est).astype(np.int)
    ys_est = np.rint(ys_est).astype(np.int)

    for idx in range(obj_list.shape[0]):
        mask_img.fill(0)
        xs = xs_est[idx]
        ys = ys_est[idx]
        for x, y in zip(xs, ys):
            cv2.circle(mask_img, (x, y), 1, 255, -1)

        # fill the holes
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
        # find contour
        contours, _ = cv2.findContours(mask_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        dst_img = cv2.drawContours(dst_img, contours, -1, get_class_colors(obj_list[idx]), 2, cv2.LINE_AA)

    return dst_img


def visualize_points(image, obj_list, pts_all, pose_est, pose_gt, k):
    # image, [h, w, 3]
    # obj_list, [n], number of object in image
    # pts_all, [cls, l, 3], model pts of all object
    # pose_est, [n, 3, 4], estimation pose
    # pose_gt, [n, 3, 4], ground truth pose
    # k, dict, camera intrinsic

    dst_img = np.copy(image)
    pts_obj = pts_all[obj_list]         # [n, l, 3]
    pts_obj = np.concatenate((pts_obj, np.ones([pts_obj.shape[0], pts_obj.shape[1], 1], dtype=np.float)), axis=2)       # [n, l, 4]
    pts_obj = pts_obj.transpose([0, 2, 1])          # [n, 4, l]
    pts_gt = np.matmul(pose_gt, pts_obj)            # [n, 3, l]
    pts_est = np.matmul(pose_est, pts_obj)          # [n, 3, l]

    xs_gt = k['fx'] * pts_gt[:, 0, :] / pts_gt[:, 2, :] + k['cx']
    ys_gt = k['fy'] * pts_gt[:, 1, :] / pts_gt[:, 2, :] + k['cy']
    xs_gt = np.rint(xs_gt).astype(np.int)
    ys_gt = np.rint(ys_gt).astype(np.int)

    for idx in range(obj_list.shape[0]):
        xs = xs_gt[idx]
        ys = ys_gt[idx]
        for x, y in zip(xs, ys):
            cv2.circle(dst_img, (x, y), 0, (0, 255, 0))

    xs_est = k['fx'] * pts_est[:, 0, :] / pts_est[:, 2, :] + k['cx']
    ys_est = k['fy'] * pts_est[:, 1, :] / pts_est[:, 2, :] + k['cy']
    xs_est = np.rint(xs_est).astype(np.int)
    ys_est = np.rint(ys_est).astype(np.int)

    for idx in range(obj_list.shape[0]):
        xs = xs_est[idx]
        ys = ys_est[idx]
        for x, y in zip(xs, ys):
            cv2.circle(dst_img, (x, y), 0, get_class_colors(obj_list[idx]+10))

    return dst_img


def visualize_vertex(image, obj_list, pts_all, pose_est, pose_gt, k):
    # image, [h, w, 3]
    # obj_list, [n], number of object in image
    # pts_all, [cls, l, 3], model pts of all object
    # pose_est, [n, 3, 4], estimation pose
    # pose_gt, [n, 3, 4], ground truth pose
    # k, dict, camera intrinsic

    line_idx = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    dst_img = np.copy(image)
    pts_obj = pts_all[obj_list]         # [n, l, 3]

    x_min, x_max = pts_obj[:, :, 0].min(axis=1), pts_obj[:, :, 0].max(axis=1)       # [n,], [n,]
    y_min, y_max = pts_obj[:, :, 1].min(axis=1), pts_obj[:, :, 1].max(axis=1)       # [n,], [n,]
    z_min, z_max = pts_obj[:, :, 2].min(axis=1), pts_obj[:, :, 2].max(axis=1)       # [n,], [n,]

    vtx = np.array([[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
                    [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]])    # [8, 3, n]
    vtx = vtx.transpose([2, 0, 1])        # [n, 8, 3]
    vtx = np.concatenate((vtx, np.ones([vtx.shape[0], 8, 1], dtype=np.float)), axis=2)      # [n, 8, 4]
    vtx = vtx.transpose([0, 2, 1])        # [n, 4, 8]

    vtx_gt = np.matmul(pose_gt, vtx)    # [n, 3, 8]
    vtx_est = np.matmul(pose_est, vtx)   # [n, 3, 8]

    xs_gt = k['fx'] * vtx_gt[:, 0, :] / vtx_gt[:, 2, :] + k['cx']
    ys_gt = k['fy'] * vtx_gt[:, 1, :] / vtx_gt[:, 2, :] + k['cy']
    xs_gt = np.rint(xs_gt).astype(np.int)
    ys_gt = np.rint(ys_gt).astype(np.int)

    for obj in range(obj_list.shape[0]):

        xs = xs_gt[obj]
        ys = ys_gt[obj]

        for idx in line_idx:
            cv2.line(dst_img, (xs[idx[0]], ys[idx[0]]), (xs[idx[1]], ys[idx[1]]), (0, 255, 0), 2)

    xs_est = k['fx'] * vtx_est[:, 0, :] / vtx_est[:, 2, :] + k['cx']
    ys_est = k['fy'] * vtx_est[:, 1, :] / vtx_est[:, 2, :] + k['cy']
    xs_est = np.rint(xs_est).astype(np.int)
    ys_est = np.rint(ys_est).astype(np.int)

    for obj in range(obj_list.shape[0]):

        xs = xs_est[obj]
        ys = ys_est[obj]

        for idx in line_idx:
            cv2.line(dst_img, (xs[idx[0]], ys[idx[0]]), (xs[idx[1]], ys[idx[1]]), get_class_colors(obj), 2)

    return dst_img


if __name__ == '__main__':

    import scipy.io as scio

    # intrinsic of camera in the YCB datasets
    K = {'fx': 1066.778, 'fy': 1067.487, 'cx': 312.9869, 'cy': 241.3109}

    img = cv2.imread('./data/000002-color.png')
    img = img[:, :, ::-1]           # BGR to RGB

    meta = scio.loadmat('./data/000002-meta.mat')
    objs = meta['cls_indexes'].flatten()
    gt_pose = meta['poses']
    gt_pose = gt_pose.transpose([2, 0, 1])
    pts = load_pts('./')

    est_pose = gt_pose.copy()
    est_pose[:, :, 3] = est_pose[:, :, 3] + np.array([0.002, 0.003, 0.01])

    res_vtx = visualize_vertex(img, objs, pts, est_pose, gt_pose, K)

    res_pts = visualize_points(img, objs, pts, est_pose, gt_pose, K)

    res_ctr = visualize_contour(img, objs, pts, est_pose, gt_pose, K)

    import matplotlib.pyplot as plt
    plt.subplot(1, 3, 1)
    plt.imshow(res_vtx)
    plt.subplot(1, 3, 2)
    plt.imshow(res_pts)
    plt.subplot(1, 3, 3)
    plt.imshow(res_ctr)

    plt.show()
