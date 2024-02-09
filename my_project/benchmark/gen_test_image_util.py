#Creating test image
from pathlib import Path
import cv2
import numpy as np
import os
import time

#*******************************************************************************************************
# pics_deliver_gen()
#*******************************************************************************************************   
def pics_deliver_gen(img_dir, max_count):
    
    index = 0
    
    for i in range(max_count):
        current_pic = img_dir / f"chunk_{index:03d}.jpg"
        index += 1
        
        if os.path.exists(current_pic):
            yield current_pic

#*******************************************************************************************************
# Class: PicsDeliver
#*******************************************************************************************************   
class PicsDeliver:
    def __init__(self, image_dir, max_count):
        self.max_count = max_count
        self.image_dir = image_dir
        self.index = 0 

    def __next__(self):
        if self.index >= self.max_count:
            raise StopIteration
        
        current_pic = self.image_dir / f"chunk_{self.index:03d}.jpg"
        #current_pic = self.img_dir + 'chunk_' + str({self.index:03d}) +'.jpg"
        #print(f'current_pic:{current_pic}')
        
        self.index += 1
        
        #if os.path.exists(current_pic):
        return current_pic
    
    def __iter__(self):
        return self
    
#*******************************************************************************************************
# Class: IterPicsDeliver
#*******************************************************************************************************   
class IterPicsDeliver:
    
    def __init__(self, image_dir, max_count):
        self.image_dir = image_dir
        self.max_count = max_count
        
    def __iter__(self):
        return PicsDeliver(self.image_dir, self.max_count)
        #return pics_deliver_gen(self.img_dir, self.max_count)

#*******************************************************************************************************
# create_test_image_chunk()
#*******************************************************************************************************   
def create_test_image_chunk(source_image_file, size):
    chunksL = []
    img_chunk_pathL = []
        
    image = cv2.imread(source_image_file, cv2.IMREAD_COLOR)
    h,w = image.shape[:2]
    
    #print(f'h = {h}, w = {w}')
    #print(f'size[0] = {size[0]}, size[1] = {size[1]}')
    #rows = int(np.ceil(image.shape[0] / size[0]))  # 行数
    #cols = int(np.ceil(image.shape[1] / size[1]))  # 列数
    
    rows = int(np.ceil(h / size[0]))  # 行数
    cols = int(np.ceil(w / size[1]))  # 列数

    #print(f'size:{size[0]}x{size[1]}')

    for row_image in np.array_split(image, rows, axis=0):
        for chunk in np.array_split(row_image, cols, axis=1):
            chunksL.append(chunk)

    #print(len(chunksL))
    
    return chunksL

#*******************************************************************************************************
# save_image_chunk()
#*******************************************************************************************************
def save_image_chunk(image_dir, image_chunksL):
    
    image_chunk_pathL = []
    
    image_dir.mkdir(exist_ok=True)

    for i, image_chunk in enumerate(image_chunksL):
        save_path = image_dir / f"chunk_{i:03d}.jpg"
        cv2.imwrite(str(save_path), image_chunk)
        image_chunk_pathL.append(str(save_path))
    
    return image_chunk_pathL

#*******************************************************************************************************
# get_image_chunk_path()
#*******************************************************************************************************
def get_image_chunk_path(image_dir, num_of_images):
    
    image_chunk_pathL = []
    
    for i in range(num_of_images):
        image_chunk_pathL.append(str(image_dir / f"chunk_{i:03d}.jpg"))
    
    return image_chunk_pathL

#*******************************************************************************************************
# remove_image_chunk()
#*******************************************************************************************************
def remove_image_chunk(image_dir, max_count):
    for res in pics_deliver_gen(image_dir, max_count):
        try:
            if os.path.exists(res):
                os.remove(res)
        except StopIteration:
            print("Stop Iteration")
        break

#*******************************************************************************************************
# pics_transaction_gen()
#*******************************************************************************************************
def pics_transaction_gen(model_instance, image_chunk_pathL, exclude_preprocessing, start_index, max_count):
    
    for i in range(max_count):
        current_pic = image_chunk_pathL[start_index]
        #print(f'current_pic:{current_pic}')
        if exclude_preprocessing:
            current_pic = model_instance._preprocessor.forward(current_pic)[0]
        start_index+=1
    
        yield current_pic

#*******************************************************************************************************
# PicsTransaction
#*******************************************************************************************************
class PicsTransaction:
    
    time_statsL = []
    
    def __init__(self, model_instance, image_chunk_pathL, exclude_preprocessing, max_count):
        self.model_instance = model_instance
        self.image_chunk_pathL = image_chunk_pathL
        self.exclude_preprocessing = exclude_preprocessing
        self.max_count = max_count
        self.index = 0 
        
    def __next__(self):
        if self.index >= self.max_count:
            raise StopIteration
        
        current_pic = self.image_chunk_pathL[self.index]
        
        if self.exclude_preprocessing:
            current_pic = self.model_instance._preprocessor.forward(current_pic)[0]
        self.index+=1
        return current_pic
    
    def __iter__(self):
        return self

#*******************************************************************************************************
# Class: IterPicsGen
#******************************************************************************************************* 
class IterPicsGen:
    
    def __init__(self, model_instance, image_chunk_pathL, exclude_preprocessing, max_count):
        self.model_instance = model_instance
        self.image_chunk_pathL = image_chunk_pathL
        self.exclude_preprocessing = exclude_preprocessing
        self.max_count = max_count
        
    def __iter__(self):
        #return PicsTransaction(self.model_instance, self.picsL, self.exclude_preprocessing, self.max_count)
        return pics_transaction_gen(self.model_instance, self.image_chunk_pathL, self.exclude_preprocessing, self.max_count)

#*******************************************************************************************************
# create_model_instances()
#******************************************************************************************************* 
def create_model_instances(model_name, zoo, batch_size, use_jpeg):

    model_instance = zoo.load_model(model_name)
    model_instance.image_backend = "pil"  # select OpenCV backend
    model_instance.input_numpy_colorspace = "RGB"      
    model_instance._model_parameters.InputImgFmt = ["JPEG" if use_jpeg else "RAW"]
    model_instance.measure_time = True
    model_instance.eager_batch_size = batch_size
    model_instance.frame_queue_depth = batch_size

    return model_instance
 
#*******************************************************************************************************
# print_result()
#******************************************************************************************************* 
def print_results(target_modelsL, infoD, image_sizeL, durationL, observed_fpsL, expected_fpsL):
    
    CW = (65,30, 16, 16)  # column widths
    header = f"| {'Model name':{CW[0]}}| {f'{image_sizeL[2]} Images Inference Time[ms]':{CW[1]}} | {'Observed FPS':{CW[2]}} | {'Max Possible FPS':{CW[3]}} |"

    print(f"hw_location:  {infoD['hw_location']}")
    print(f"model_zoo_url:{infoD['model_zoo_url']}")
    print(f"Batch Size:   {infoD['batch_size']}")
    print(f'Image Size:   {image_sizeL[0]} x {image_sizeL[1]}')
    print(f"{'-'*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    for mi, model in enumerate(target_modelsL):
        print(f"| {target_modelsL[mi]:{CW[0]}}|" + f" {durationL[mi]:{CW[1]}} |" + f" {observed_fpsL[mi]:{CW[2]}} |" + f" {expected_fpsL[mi]:{CW[3]}} |")
    
    print(f"{'-'*len(header)}")    
    
#*******************************************************************************************************
# save_result()
#******************************************************************************************************* 
def save_results(target_modelsL, infoD, image_sizeL, durationL, observed_fpsL, expected_fpsL, sz):
    CW = (65,30, 16, 16)  # column widths
    header = f"| {'Model name':{CW[0]}}| {f'{image_sizeL[2]} Images Inference Time[ms]':{CW[1]}} | {'Observed FPS':{CW[2]}} | {'Max Possible FPS':{CW[3]}} |"

    #d = time.strftime('%m%d%H%M')
    data_dir = '/home/gotom/data_dir/200pics_measure/' 
    file = data_dir + 'mobile_net_test_results.txt'

    if sz == 0:
        f = open(file, 'w')
        
    else: 
        f = open(file, 'a')
    
    f.write(f"hw_location:  {infoD['hw_location']}\n")
    f.write(f"model_zoo_url:{infoD['model_zoo_url']}\n")
    f.write(f"Batch Size:   {infoD['batch_size']}\n")
    f.write(f'Image Size:   {image_sizeL[0]} x {image_sizeL[1]}\n')
    f.write(f"{'-'*len(header)}\n")
    f.write(header + '\n')
    f.write(f"{'-'*len(header)}\n")

    for mi, model in enumerate(target_modelsL):
    
        f.write(f"| {target_modelsL[mi]:{CW[0]}}|" + f" {durationL[mi]:{CW[1]}} |" + f" {observed_fpsL[mi]:{CW[2]}} |" + f" {expected_fpsL[mi]:{CW[3]}} |\n")
    
    f.write(f"{'-'*len(header)}\n")
    
    if sz == 2:
        d = time.strftime('%m%d%H%M')
        os.rename(file, data_dir + 'mobile_net_test_results' + d + '.txt')
    f.close()