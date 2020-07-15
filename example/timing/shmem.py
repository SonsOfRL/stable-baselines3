import multiprocessing
import ctypes
import numpy as np
from multiprocessing.sharedctypes import RawArray
import time


N_PROCS = 4
ARRAY_SIZE = 10


def worker(rank, shm):
    np_shm = np.frombuffer(shm, dtype=np.float32)
    np_shm[rank*2: (rank+1)*2] = rank
    time.sleep(0.1)
    print("Rank: {}, array: {}".format(rank, np_shm))


def main():

    # -- <SHM --
    shm = RawArray(ctypes.c_float, ARRAY_SIZE)
    np_shm = np.frombuffer(shm, dtype=np.float32)
    # -- SHM> --

    np_shm[:] = np.arange(10)

    processes = []
    for rank in range(N_PROCS):
        proc = multiprocessing.Process(
            target=worker,
            args=(rank, shm),
            daemon=True,
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    print("Rank: {}, array: {}".format("master", np_shm))
    


if __name__ == "__main__":
    main()
