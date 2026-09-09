import os
import numpy as np
import pydicom 
import SimpleITK as sitk 
import datetime
import pickle
from tqdm import tqdm 
import time
def save_itk(data, path):
    sitk.WriteImage(sitk.GetImageFromArray(data), path)
def load_itk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))
def load_bin(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
def save_bin(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def mkdir(p, is_file = False):
    

    if is_file:
        result, _ = os.path.split(p)
        
    else:
        result = p
    isExists = os.path.exists(result)
    if isExists:
        pass
    else:
        os.makedirs(result)
        print("make directory successfully:{}".format(result))

def load_pet_suv(path):
    
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) 
    # import pdb 
    # pdb.set_trace() 
    arr = []
    for s in slices:
        img = s.pixel_array 
        a=s.RescaleSlope
        b=s.RescaleIntercept 
        img = a*img + b 
        arr.append(img) 

    arr = np.stack(arr)
    
    info = slices[0] 

    activity = info.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    weight = info.PatientWeight 
    # import pdb 
    # pdb.set_trace()
    injectionTime = info.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    injectionTime = datetime.datetime.strptime(str(int(float(injectionTime))),"%H%M%S")
    
    acqTime = info.AcquisitionTime
    acqTime = datetime.datetime.strptime(str(int(float(acqTime))),"%H%M%S")
    
    DecayCorrection = info.DecayCorrection
    
    if DecayCorrection == " ADMIN":
        duration = 0
    elif DecayCorrection == "START":
        duration = (acqTime - injectionTime).total_seconds()
    else:
        raise Exception("Unknown DecayCorrection")
    # import pdb 
    # pdb.set_trace()
    halflife = info.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    a=info.RescaleSlope
    b=info.RescaleIntercept
    
    scale = 2**(duration/halflife) * weight * 1000 / activity
    
    result = arr*scale
    
    return np.float32(result) 

def load_ct_hu(path):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial 
    
    '''
    get slices from dicom
    '''
    
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness 
        
    '''
    hu image
    '''

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16) 

def process_image(file_name, src_path, save_path):

    src_path = os.path.join(src_path, file_name, 'lung', "LQ") 
    target_path = os.path.join(save_path, "LQ", "{}.nii.gz".format(file_name)) 
    

    try: 

        data = load_ct_hu(src_path) 
        save_itk(data, target_path)

    except Exception as e:
        print(f"Failed to process {file_name}: {e}")

# since = time.time()
# data = load_itk(r"D:\Dataset\CT\CT\HQ\L291.nii.gz")
# print(time.time()-since)


# '''
# aapm
# '''

# root = r"D:\Dataset\CT\FD_1mm_sharp\full_1mm_sharp" 
# save_path = r"D:\Dataset\CT\CT\train-aapm"
# file_list = os.listdir(root) 
# for f in tqdm(file_list):
#     data = load_ct_hu(os.path.join(root, f, "full_1mm_sharp")) 
#     save_itk(data, os.path.join(save_path, "HQ", "{}.nii.gz".format(f)))
    


# '''
# LDCT
# '''
# from concurrent.futures import ThreadPoolExecutor, as_completed
# root = r"F:\Dataset\All-in-One-Large\3D\CT\LDCT" 
# save_path = r"D:\Dataset\CT\CT"
# file_list = os.listdir(root) 

# with ThreadPoolExecutor(max_workers=4) as executor: 
#     futures = [executor.submit(process_image, file_name, root, save_path) for file_name in file_list] 
#     for future in as_completed(futures):
#         future.result() 

# '''
# sino
# '''
# from concurrent.futures import ThreadPoolExecutor, as_completed
# root = r"F:\Dataset\All-in-One-Large\3D\CT\sino\sino\2_abdomen" 
# save_path = r"D:\Dataset\CT\CT\train-sino-abdomen"
# file_list = os.listdir(root) 

# with ThreadPoolExecutor(max_workers=4) as executor: 
#     futures = [executor.submit(process_image, file_name, root, save_path) for file_name in file_list] 
#     for future in as_completed(futures):
#         future.result()


'''
sino
'''
from concurrent.futures import ThreadPoolExecutor, as_completed
root = r"D:\Dataset\CT\Lung\Lung\TM_Lung_1.25mm" 
save_path = r"D:\Dataset\CT\CT\train-sino-lung"
mkdir(os.path.join(save_path, "HQ")) 
mkdir(os.path.join(save_path, "LQ"))
file_list = os.listdir(root) 

with ThreadPoolExecutor(max_workers=8) as executor: 
    futures = [executor.submit(process_image, file_name, root, save_path) for file_name in file_list] 
    for future in as_completed(futures):
        future.result()
    

    