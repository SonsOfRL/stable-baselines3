import random
import time
import multiprocessing as mp


def worker(rank, job_queue, ready_queue):
    total_call = 0
    job_waiting = 0
    job_processing = 0
    while True:
        s_t = time.time()
        ix = job_queue.get()
        job_waiting += time.time() - s_t

        s_t = time.time()
        if ix == -1:
            break
        if random.random() < 0.95:
            time.sleep(0.01)
        else:
            time.sleep(0.5)
        job_processing += time.time() - s_t
        total_call += 1
        ready_queue.put(ix)
    print("Process: {}, Total call: {}, Waiting: {}: Processing: {}".format(
        rank, total_call, job_waiting, job_processing))


if __name__ == "__main__":

    n_iters = int(1e2)
    n_procs = 8
    n_job_per_proc = 8
    n_batch = 8
    job_index_queue = [mp.Queue(n_job_per_proc) for i in range(n_procs)]
    ready_index_queue = mp.Queue(n_procs * n_job_per_proc)

    batch_waiting = 0

    job_proc_map = {i: i // n_job_per_proc
                    for i in range(n_job_per_proc * n_procs)}
    proc_job_map = {i: list(range(i * n_job_per_proc, (i + 1) * n_job_per_proc))
                    for i in range(n_procs)}

    processes = []
    for i in range(n_procs):
        process = mp.Process(
            target=worker,
            args=(i, job_index_queue[i], ready_index_queue)
        )
        process.start()
        processes.append(process)

    # Start jobs
    batch_ix = list(range(n_job_per_proc * n_procs))

    # Iterate
    total_time_s = time.time()
    for j in range(n_iters):

        for ix in batch_ix:
            job_index_queue[job_proc_map[ix]].put(ix)

        s_t = time.time()
        batch_ix = [ready_index_queue.get() for i in range(n_batch)]
        e_t = time.time()
        batch_waiting += (e_t - s_t)

        if random.random() < 0.9:
            time.sleep(random.random() * 0.02 + 0.01)
        else:
            time.sleep(0.1)

        print("Iteration: {}".format(j), end="\r")
    total_time_e = time.time()

    print(" " * 80)
    print("Master waiting time: {}, total time: {}".format(
        batch_waiting, total_time_e - total_time_s))

    # Finalize jobs
    while ready_index_queue.empty() is False:
        ready_index_queue.get()
        # print("unused: ", ready_index_queue.get())
    for queue_ix, ixs in proc_job_map.items():
        job_index_queue[queue_ix].put(-1)

    for p in processes:
        p.join()
