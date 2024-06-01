from skimage import metrics

def find_HD(y, y_pred):     
    labeled_pred, count_pred = label(y_pred, return_num = True)
    pred_objects = regionprops(labeled_pred)
    pred_object_coord = [obj["coords"] for obj in pred_objects] 


    labeled_gt, count_gt = label(y, return_num = True)
    true_objects = regionprops(labeled_gt)
    true_object_coord = [obj["coords"] for obj in true_objects] 

    lesion_images_gt = []
    for elem in true_object_coord:
        lesion_image = np.zeros_like(labeled_gt)
        for coordinate in elem:
            lesion_image[coordinate[0], coordinate[1]] = 1

        lesion_images_gt.append(lesion_image)


    lesion_images_pred = []
    for elem in pred_object_coord:
        lesion_image = np.zeros_like(labeled_gt)
        for coordinate in elem:
            lesion_image[coordinate[0], coordinate[1]] = 1

        lesion_images_pred.append(lesion_image)

    HD_list = []
    counter = 0
    for elem_pred in lesion_images_pred:
        for elem_gt in lesion_images_gt:
            if np.sum(elem_pred*elem_gt > 0):
                HD = metrics.hausdorff_distance(elem_pred, elem_gt)
                HD_list.append(HD)
                counter += 1
    return HD_list
