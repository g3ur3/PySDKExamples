import numpy as np
import degirum as dg, dgtools
import matplotlib.pyplot as plt
#import mytools
import threading

#*******************************************************************************************************
# create_model_instances()
#*******************************************************************************************************
def create_model_instances(model_names, zoo_names, use_jpeg):

    model_instances = []
    #zoo = {}

    for mi, model_name in enumerate(model_names):
        model = zoo_names[mi].load_model(model_name)
        #model.image_backend = "opencv"  # select OpenCV backend
        #model.input_numpy_colorspace = "BGR"      
        model.image_backend = "pil"  # select OpenCV backend
        model.input_numpy_colorspace = "RGB"      
        model._model_parameters.InputImgFmt = ["JPEG" if use_jpeg else "RAW"]
        model.measure_time = True
        model_instances.append(model)

    return model_instances

#*******************************************************************************************************
# baseline_test()
#*******************************************************************************************************
def baseline_test(model_name, model_instance, image_file, \
                  exclude_preprocessing, iterations, batch):
    
    time_stats = {}
    model_instance.eager_batch_size = batch
    model_instance.frame_queue_depth = batch

    frame = image_file 
    if exclude_preprocessing:
        frame = model_instance._preprocessor.forward(frame)[0]

    def source():
        for fi in range(iterations):
            yield frame

    model_instance(frame)  # run model once to warm up the system
    model_instance.reset_time_stats()

    # batch predict: measure throughput
    t = dgtools.Timer()

    for res in model_instance.predict_batch(source()):
        pass  

    time_stats[model_name] = model_instance.time_stats()

    ret = {
            "elapsed": t(),
            "time_stats": time_stats,
        }

    return ret

#*******************************************************************************************************
# multi_models_series_test()
#*******************************************************************************************************
def multi_models_series_test(model_names, model_instances, image_file, \
                             exclude_preprocessing, iterations, batch):
    inference_results = {}
    time_stats =  {}
    data = []

    for mi, model_name in enumerate(model_names):
        inference_results[model_name] = {}
        time_stats[model_name] = {}

    for mi, model in enumerate(model_instances):   
        frame = image_file      
        if exclude_preprocessing:
            frame = model._preprocessor.forward(frame)[0]
        data.append(frame)  
    # define source of frames
    def source(mi):
        for fi in range(iterations):
            yield data[mi]

    for mi, model in enumerate(model_instances):
        model.eager_batch_size = batch
        model.frame_queue_depth = batch
        model(data[mi])  # run model once to warm up the system
        model.reset_time_stats()
    
    t = dgtools.Timer()
    for mi, model in enumerate(model_instances):
        for res in model.predict_batch(source(mi)):
            pass
        time_stats[model_names[mi]] = model.time_stats() 
        inference_results[model_names[mi]] = res._inference_results
        
    return {
        "elapsed": t(),
        "time_stats": time_stats,
        "inference_results": inference_results,
    }

#*******************************************************************************************************
# multi_model_parallel_test()
#*******************************************************************************************************
def multi_models_parallel_test(model_names, model_instances, image_file,\
                                exclude_preprocessing, iterations, batch):
    ret = {}
    data = []   
    nmodels = len(model_instances)
    for mi, model_name in enumerate(model_names):
        ret[model_name] = {}
        frame = image_file
        if exclude_preprocessing:
            frame = model_instances[mi]._preprocessor.forward(frame)[0]
        data.append(frame)
    # define source of frames
    def source(mi):
        for fi in range(iterations):
            yield data[mi]  
    barr = threading.Barrier(nmodels)

    def run_one_model(mi):
        with model_instances[mi] as model:
            model.eager_batch_size = batch
            model.frame_queue_depth = batch
            model(data[mi])  # run model once to warm up the system
            model.reset_time_stats()
            barr.wait()
            t = dgtools.Timer()          
            for res in model.predict_batch(source(mi,)):
                pass
            ret[model_names[mi]] = {
                    "elapsed": t(),
                    "time_stats": model.time_stats(),
                }
    threads = [
            threading.Thread(target=run_one_model, args=(mi,)) for mi in range(nmodels)
        ]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]

    return ret

#*******************************************************************************************************
# draw_graph()
#*******************************************************************************************************
def draw_graph(data_dir, model_name, model_symbol, batch_value, observed_fps, expected_fps, device_temp, y_max, suffix):
    
    t = batch_value
    observed_fps_values = []
    expected_fps_values = []
    device_temp_values = []

    for k, v in observed_fps.items():
        observed_fps_values.append(v)
    
    for k, v in expected_fps.items():
        expected_fps_values.append(v)

    for k, v in device_temp.items():
        device_temp_values.append(v)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    plt.rcParams["font.family"] = "Dejavu Serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["lines.linewidth"] = 1.5

    plt.grid()
    font = {'family' : 'Dejavu Serif',
            'size' : 11,
    }

    ax1.set_ylim(0, y_max)
    ax1.plot(t, expected_fps_values, color = 'yellowgreen', label = 'Expected FPS', marker = 'D', markersize = 4, 
             markeredgewidth = 1.5, markeredgecolor = 'lime', markerfacecolor = 'yellowgreen')
    
    for i, value in enumerate(expected_fps_values):
        ax1.text(t[i], expected_fps_values[i], value, size = 10, ha = 'left', va = 'bottom', color = 'black')
        
    ax1.plot(t, observed_fps_values, color = 'lightblue', label = 'Observed FPS',marker = 'D', markersize = 4, 
             markeredgewidth = 1.5, markeredgecolor = 'skyblue', markerfacecolor = 'lightblue')
    
    for i, value in enumerate(observed_fps_values):
        ax1.text(t[i], observed_fps_values[i], value, size = 10, ha = 'right', va = 'top', color = 'black')
    
    ax1.set_xlabel('Batch size', fontdict=font)
    ax1.set_ylabel('FPS', fontdict=font)

    ax1.legend(loc = 'upper center')

    # ax2 = ax1.twinx()
    # ax2.set_ylim(40, 100)
    # ax2.plot(t, device_temp_values, color = 'red', label = 'Temperature(℃)' )
    # ax2.set_ylabel('Temperature(C)')

    
    plt.title(model_name)
    
    plt.show()
    fig.savefig(data_dir + model_symbol +'_' + suffix + '.jpg')

#*******************************************************************************************************
# show_results()
#*******************************************************************************************************
def show_results(items_dic, expected_fps, observed_fps):
    
    print(f'mode:            {items_dic["mode"]}')   
    print(f'Device:          {items_dic["Device"]}')
    print(f'iterations:      {items_dic["iterations"]}')
    print(f'model_name1:     {items_dic["model_name1"]}')
    print(f'model_name2:     {items_dic["model_name2"]}')    
    print(f'image_file_name: {items_dic["image_file_name"]}')
    print(f'expected_fps:    {expected_fps}')
    print(f'observed_fps:    {observed_fps}')
    print('')

#*******************************************************************************************************
# save_results()
#*******************************************************************************************************  
def save_results(file_obj, items_dic, expected_fps, observed_fps):
    
    file_obj.write(f'Mode:            {items_dic["mode"]}\n')
    file_obj.write(f'Device:          {items_dic["Device"]}\n')
    file_obj.write(f'iterations:      {items_dic["iterations"]}\n')
    file_obj.write(f'model_name1:     {items_dic["model_name1"]}\n')
    file_obj.write(f'model_name2:     {items_dic["model_name2"]}\n')
    file_obj.write(f'image_file_name: {items_dic["image_file_name"]}\n')
    file_obj.write(f'expected fps:    {expected_fps}\n')
    file_obj.write(f'observed fps:    {observed_fps}\n')
    file_obj.write('\n')
    file_obj.close()

#*******************************************************************************************************
# pics_transaction_gen()
#*******************************************************************************************************     
def pics_transaction_gen(model_instance, picsL, exclude_preprocessing, max_count):
    
    index = 0
    
    for i in range(max_count):
        current_pic = picsL[index]
        if exclude_preprocessing:
            current_pic = model_instance._preprocessor.forward(current_pic)[0]
        index+=1
    
        yield current_pic

#*******************************************************************************************************
# Class: PicsTransaction
#*******************************************************************************************************   
class PicsTransaction:
    
    picsL = []
    time_statsL = []
    def __init__(self, model_instance, picsL, exclude_preprocessing, max_count):
        self.model_instance = model_instance
        self.picsL = picsL
        self.exclude_preprocessing = exclude_preprocessing
        self.max_count = max_count
        self.index = 0 
        
    def __next__(self):
        if self.index >= self.max_count:
            raise StopIteration
        
        current_pic = self.picsL[self.index]
        
        if self.exclude_preprocessing:
            current_pic = self.model_instance._preprocessor.forward(current_pic)[0]
        self.index+=1
        return current_pic
    
    def __iter__(self):
        return self
    
#*******************************************************************************************************
# Class: MyIteration
#*******************************************************************************************************  
class MyIteration:
    
    def __init__(self, model_instance, picsL, exclude_preprocessing, max_count):
        self.model_instance = model_instance
        self.picsL = picsL
        self.exclude_preprocessing = exclude_preprocessing
        self.max_count = max_count
        
    def __iter__(self):
        return PicsTransaction(self.model_instance, self.picsL, self.exclude_preprocessing, self.max_count)
        #return pics_transaction_gen(self.model_instance, self.picsL, self.exclude_preprocessing, self.max_count)