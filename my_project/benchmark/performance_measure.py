import numpy as np
import performance_measure_util as putil
import time

#*******************************************************************************************************
# temperature_detect()
#*******************************************************************************************************
def operating_temperature_verify(temperature):
    while (temperature > 79):
        print('Threshold exceeded!!')
        time.sleep(60)

#*******************************************************************************************************
# base_line_fps_measure()
#*******************************************************************************************************
def baseline_fps_measure(data_dir, test_cycle, iterations, model_name, model_instance, image_file, \
                         exclude_preprocessing, batch, model_symbol, suffix, arr_save):

    device_fps = []
    observed_fps = []
    expected_fps = []
    device_temperature = []

    for i in range(test_cycle):
        
        res = putil.baseline_test(model_name, model_instance, image_file, \
                                  exclude_preprocessing, iterations, batch)
        
        inference_ms = res["time_stats"][model_name]["CoreInferenceDuration_ms"].avg
        #device_inference_ms = res["time_stats"][model_name]['DeviceInferenceDuration_ms'].avg
        frame_duration_ms = 1e3 * res["elapsed"] / iterations
        #device_temperature_value = res["time_stats"][model_name]["DeviceTemperature_C"].max

        #device_fps_value = round(1e3 / device_inference_ms, 1)
        expected_fps_value = round(1e3 / inference_ms, 1)
        observed_fps_value = round(1e3 / frame_duration_ms, 1) 

        #device_fps.append(device_fps_value)        
        observed_fps.append(observed_fps_value)
        expected_fps.append(expected_fps_value)
        #device_temperature.append(device_temperature_value)

        #operating_temperature_verify(device_temperature_value)

    #device_fps_arr = np.array(device_fps)
    observed_fps_arr = np.array(observed_fps)
    expected_fps_arr = np.array(expected_fps)
    #device_temp_arr = np.array(device_temperature)

    #device_fps_avg = round(np.mean(device_fps_arr),1)
    observed_fps_avg = round(np.mean(observed_fps_arr),1)
    expected_fps_avg = round(np.mean(expected_fps_arr),1)
    #device_temp_max = round(np.max(device_temp_arr), 1)

    if arr_save:
        #np.save(data_dir + model_symbol + '_device_fps_' + suffix + '.npy', device_fps_arr)
        np.save(data_dir + model_symbol + '_observed_fps_' + suffix + '.npy', observed_fps_arr)
        np.save(data_dir + model_symbol + '_expected_fps_' + suffix + '.npy', expected_fps_arr)
        #np.save(data_dir + model_symbol + '_device_temp_' + suffix + '.npy', device_temp_arr)

    #test_results = {'observed_fps': observed_fps_avg, 'expected_fps': expected_fps_avg, 'device_fps': device_fps_avg, 'device_temp': device_temp_max}
    test_results = {'observed_fps': observed_fps_avg, 'expected_fps': expected_fps_avg}

    return test_results

#*******************************************************************************************************
# series_models_fps_measure()
#*******************************************************************************************************
def series_models_fps_measure(data_dir, test_cycles, iterations, model_names, model_instances, image_file, \
                              exclude_preprocessing, batch, model_symbol, suffix, arr_save):

    observed_fps = []
    expected_fps = []
    device_temperature = []
    
    for i in range(test_cycles):
        res = putil.multi_models_series_test(model_names, model_instances, \
                                        image_file, exclude_preprocessing, iterations, batch)
        
        inference_ms = 0
    
        for im in range(len(model_names)):
            inference_ms += res['time_stats'][model_names[im]]["CoreInferenceDuration_ms"].avg   
        frame_duration_ms = 1e3 * res["elapsed"] / iterations       
        observed_fps_value = round(1e3 / frame_duration_ms, 1)
        expected_fps_value = round(1e3 / inference_ms, 1)
        
        device_temperature_value = res['time_stats'][model_names[im]]["DeviceTemperature_C"].max
        
        observed_fps.append(observed_fps_value)
        expected_fps.append(expected_fps_value)
        device_temperature.append(device_temperature_value)

        operating_temperature_verify(device_temperature_value)

    observed_fps_arr = np.array(observed_fps)
    expected_fps_arr = np.array(expected_fps)
    device_temp_arr = np.array(device_temperature)
       
    observed_fps_avg = round(np.mean(observed_fps_arr),1)
    expected_fps_avg = round(np.mean(expected_fps_arr),1)
    device_temp_max = round(np.max(device_temp_arr),1)
    
    if arr_save:
        np.save(data_dir + model_symbol + '_observed_fps_' + suffix + '.npy', observed_fps_arr)
        np.save(data_dir + model_symbol + '_expected_fps_' + suffix + '.npy', expected_fps_arr)
        np.save(data_dir + model_symbol + '_device_temp_' + suffix + '.npy', device_temp_arr)

    test_results = {'observed_fps':observed_fps_avg, 'expected_fps':expected_fps_avg, 'device_temp': device_temp_max}

    return test_results

#*******************************************************************************************************
# parallel_models_fps_measure()
#*******************************************************************************************************
def parallel_models_fps_measure(data_dir, test_cycles, iterations, model_names, model_instances,\
                                 image_file, exclude_preprocessing, batch, model_symbol, suffix, arr_save):

    observed_fps = []
    expected_fps = []
    device_temperature = []

    for i in range(test_cycles):
        
        res = putil.multi_models_parallel_test(model_names, model_instances, image_file,\
                                            exclude_preprocessing, iterations, batch)
    
        inference_ms = 0
        frame_duration_ms = 0
        
        for im in range(len(model_names)):
            inference_ms = max(inference_ms, res[model_names[im]]["time_stats"]["CoreInferenceDuration_ms"].avg)
            frame_duration_ms = max(frame_duration_ms, 1e3 * res[model_names[im]]["elapsed"] / iterations)           
            device_temperature_value = res[model_names[im]]["time_stats"]["DeviceTemperature_C"].max
        observed_fps_value = round(1e3 / frame_duration_ms, 1)
        expected_fps_value = round(1e3 / inference_ms, 1)
        
        observed_fps.append(observed_fps_value)
        expected_fps.append(expected_fps_value)
        device_temperature.append(device_temperature_value)

        #operating_temperature_verify(device_temperature_value)
        
    observed_fps_arr = np.array(observed_fps)
    expected_fps_arr = np.array(expected_fps)
    device_temp_arr = np.array(device_temperature)

    observed_fps_avg = round(np.mean(observed_fps_arr),1)
    expected_fps_avg = round(np.mean(expected_fps_arr),1)
    device_temp_max = round(np.max(device_temp_arr),1)

    test_results = {'observed_fps':observed_fps_avg, 'expected_fps':expected_fps_avg, 'device_temp': device_temp_max}
    
    if arr_save:
        np.save(data_dir + model_symbol + '_observed_fps_' + suffix + '.npy', observed_fps_arr)
        np.save(data_dir + model_symbol + '_expected_fps_' + suffix + '.npy', expected_fps_arr)
        np.save(data_dir + model_symbol + '_device_temp_' + suffix + '.npy', device_temp_arr)

    return test_results

#*******************************************************************************************************
# two_parallel_models_fps_measure()
#*******************************************************************************************************
def two_parallel_models_fps_measure(data_dir, test_cycles, iterations, model_names, model_instances,\
                                 image_file, exclude_preprocessing, batch, model_symbol, suffix, arr_save):

    observed_fps =[[]]
    
    for i in range(test_cycles):
        
        res = putil.multi_models_parallel_test(model_names, model_instances, image_file,\
                                            exclude_preprocessing, iterations, batch)
    
        frame_duration_ms = 0
        
        for im in range(len(model_names)):
            frame_duration_ms = res[model_names[im]]["elapsed"] / iterations
            
            observed_fps_value = round(1e3 / frame_duration_ms, 1)
        
            observed_fps[im].append(observed_fps_value)
                
        observed_fps_arr_0 = np.array(observed_fps[0])
        observed_fps_arr_1 = np.array(observed_fps[1])
   
    observed_fps_0_avg = round(np.mean(observed_fps_arr_0),1)
    observed_fps_1_avg = round(np.mean(observed_fps_arr_1),1)
   
    test_results = {'observed_fps' + model_names[0]:observed_fps_0_avg, 'observed_fps' + model_names[1]:observed_fps_1_avg}
    
    if arr_save:
        np.save(data_dir + model_symbol + '_observed_fps_0_' + suffix + '.npy', observed_fps_arr_0)
        np.save(data_dir + model_symbol + '_observed_fps_1_' + suffix + '.npy', observed_fps_arr_1)
        
    return test_results

