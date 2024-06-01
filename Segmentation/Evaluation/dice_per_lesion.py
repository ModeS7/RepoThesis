from skimage.measure import label, regionprops
import numpy as np
def dice_per_lesion(y, y_pred):
    #label true and predicted segmentation masks
    labeled_image,count = label(y, return_num=True)
    labeled_image_pred,count_pred = label(y_pred, return_num=True)
    
    true_objects = regionprops(labeled_image)
    true_object_coord = [obj["coords"] for obj in true_objects] 
    
    pred_objects = regionprops(labeled_image_pred)
    pred_object_coord = [obj["coords"] for obj in pred_objects]
    
    dice_per_lesion_list = []
    
    for coordinates_true in true_object_coord:
        for coordinates_pred in pred_object_coord: 
            all_cordinates = []
            all_cordinates.append(coordinates_pred)
            all_cordinates.append(coordinates_true)
            all_cordinates = np.concatenate(all_cordinates, axis = 0)
    
            all_cordinates_tuple = []
            for elem in all_cordinates:
                elem_tuple = tuple(elem)
                all_cordinates_tuple.append(elem_tuple)
            all_cordinates_set = set(all_cordinates_tuple)
            
            detected = False
            
            if len(all_cordinates_set) != len(all_cordinates_tuple):
                detected = True
            
            if detected:
                TP = 0
                FP = len(coordinates_pred)
                FN = len(coordinates_true)
                
                for true_coordinate in coordinates_true:
                    true_coordinate_tuple = tuple(true_coordinate)
                    for pred_coordinate in coordinates_pred:
                        pred_coordinate_tuple = tuple(pred_coordinate)
                        
                        if pred_coordinate_tuple == true_coordinate_tuple:
                            TP += 1
                            FN -= 1
                            FP-= 1
                        
                            
                dice = 2*TP /(2*TP + FP + FN)
                dice_per_lesion_list.append(dice)
                break
    
    return dice_per_lesion_list
