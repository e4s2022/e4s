
import cv2
import numpy as np


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_


def blending(full_img, ori_img, mask):
    height, width = ori_img.shape[:2]

    mask_sharp = 1
    
    """
    try:
        new_h = 2 ** (int(np.log2(height)) + 1)
        new_w = 2 ** (int(np.log2(width)) + 1)
        full_img, ori_img, full_mask = [cv2.resize(x, (new_h, new_w)) for x in (full_img, ori_img, np.float32(mask_sharp * mask))]
        # full_img = cv2.convertScaleAbs(ori_img*(1-full_mask) + full_img*full_mask)
        img = Laplacian_Pyramid_Blending_with_mask(full_img, ori_img, full_mask, 10)
    except:
    """
    new_h = 1024
    new_w = 1024
    full_img, ori_img, full_mask = [cv2.resize(x, (new_h, new_w)) for x in (full_img, ori_img, np.float32(mask_sharp * mask))]
    # full_img = cv2.convertScaleAbs(ori_img*(1-full_mask) + full_img*full_mask)
    img = Laplacian_Pyramid_Blending_with_mask(full_img, ori_img, full_mask, 10)

    ### img in [0, 255]
    img = np.clip(img, 0 ,255)
    img = np.uint8(cv2.resize(img, (width, height)))
    return img
