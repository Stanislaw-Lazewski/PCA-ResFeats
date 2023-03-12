from time import sleep, time
from random import random
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import csv


def progress_indicator(future):
    global lock, tasks_total, tasks_completed, t
    with lock:
        tasks_completed += 1
        # report progress
        print(f'\r{100*tasks_completed/tasks_total}%\t Time to the end [min]: {(tasks_total-tasks_completed)*(time()-t)/(60*tasks_completed)}', end="")


def task(f_lock, file_handle, classification_params, data_list, task_function):
    accuracy = task_function(*classification_params)
    if isinstance(accuracy, list):
        with f_lock:
            writer = csv.writer(file_handle) 
            for variant_result in accuracy:
                writer.writerow(data_list+variant_result)
    else:
        data_list.append(accuracy)
        with f_lock:
            writer = csv.writer(file_handle)                                
            writer.writerow(data_list)
        
def multithread_task(data_params, result_path, n, task_function):
    global t
    t=time()
    # A lock for the counter
    global lock
    lock = Lock()
    # Total tasks we will execute
    global tasks_total
    tasks_total = n
    # Total completed tasks
    global tasks_completed
    tasks_completed = 0
    with open(result_path, 'a', encoding='UTF8') as handle:
        # A lock to protect the file
        f_lock = Lock()
        with ThreadPoolExecutor(120) as executor:
            futures = [executor.submit(task, f_lock, handle, classification_params, data_list, task_function) for (classification_params, data_list) in data_params]
            
            for future in futures:
                future.add_done_callback(progress_indicator)
        print('\nDone!')
        
