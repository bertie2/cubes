import multiprocessing

chunk_size = 10000
manager = None
logging_queue = None
pool = None

def logging_task(queue):
    try:
        status = {}
        while True:
            (uid, done, total) = queue.get()
            status[uid] = (done, total)

            total_done = 0
            total_total = 0
            for uid, (done, total) in status.items():
                total_done += done
                total_total += total

            if(total_done == total_total):
                print(f'\rCompleted {total_done} of {total_total} {((total_done/total_total) * 100):.2f}%')
                status = {}
            else:
                print(f'\rCompleted {total_done} of {total_total} {((total_done/total_total) * 100):.2f}%', end="")
    except:
        print("multithreaded logging ended")

def init_parrelism():
    global logging_queue
    global pool
    global manager 
    manager = multiprocessing.Manager()
    logging_queue = manager.Queue()
    logging_process = multiprocessing.Process(target=logging_task, args=[logging_queue])
    logging_process.daemon = True
    logging_process.start()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

def dispatch_tasks(task_function, items, enable_multicore):
    #todo dont keep everything in memory, write out to files async as we go
    if(enable_multicore):
        chunks = []
        for chunk_base in range(0, len(items), chunk_size):
            chunks.append((items[chunk_base: min( chunk_base + chunk_size, len(items))], logging_queue))
        items = None
        results = pool.map(task_function, chunks)
        final_result = []
        for result in reversed(results):
            if(isinstance(result, list)):
                final_result += result
            else:
                final_result += list(result)
        return final_result
    else:
        return task_function((items, None))
