import multiprocessing
import ctypes
from collections import OrderedDict, namedtuple
from typing import Union, Type, Optional, Dict, Any, List, Tuple, Callable, Sequence
import gym
from gym.spaces import Box, Discrete
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper


def _worker():
    """ Environment running function. Run multiple environment each with an
    associated index. Worker communicate with the master process via two
    queues namely, ready queue and job queue. When an index arrives into job
    queue, worker process run the environment with the index. When the
    environment is done stepping, worker process pushes the index into the
    ready queue. Environemnts are run in series within a worker process. In
    environment step, shared memory partition pointed by the index is first
    read to obtain action information and written back with the data that is
    returned by the environment.
    """
    pass


class SharedMemoryGenerator():

    DtypeMap = {
        np.float64: ctypes.c_double,
        np.float32: ctypes.c_float,
        np.int64: ctypes.c_int64,
        np.int32: ctypes.c_int32,
        np.uint8: ctypes.c_uint8,
        np.bool: ctypes.c_bool,
    }

    EnvInfo = namedtuple("EnvInfo", "obs act")
    DataInfo = namedtuple("DataInfo", "shape dtype")
    SharedRollouts = namedtuple("SharedRollouts", "obs, act, reward, done")
    StepTriple = namedtuple("StateTriple", "obs, reward, done")

    def __init__(self, env_fn, n_env):

        self.n_env = n_env
        self.env_spaces = self.EnvInfo(env.observation_space, env.action_space)

        if isinstance(env.action_space, Box):
            act_shape = env.action_space.shape
            act_dtype = env.action_space.dtype
        elif isinstance(env.action_space, Discrete):
            act_shape = (1,)
            act_dtype = np.int64
        else:
            raise ValueError("Action space is unknown")

        self.envinfo = self.EnvInfo(
            self.DataInfo(
                env.observation_space.shape,
                enb.observation_space.dtype
            ),
            self.DataInfo(
                act_shape,
                act_dtype
            )
        )

    def make_shared_mems(self):

        shared_mems = []
        buffers = []
        for shape, dtype in (*self.envinfo, ((1,), np.float64), ((1,), np.bool)):
            shm = multiprocessing.sharedctypes.RawArray(
                self.DtypeMap[dtype],
                np.prod(shape) * self.n_env
            )
            shared_mems.append(shm)
            buffers.append(np.frombuffer(shm, dtype=dtype))

        self.buffer = self.SharedRollouts(*buffers)
        return self.SharedRollouts(*shared_mems)

    def __getitem__(self, nd_indexes: np.ndarray):
        """ params: nd_indexes: 1D int numpy array of indexes """
        return self.StepTriple(
            self.buffer.obs[nd_indexes],
            self.buffer.reward[nd_indexes],
            self.buffer.done[nd_indexes]
        )

    def __setitem__(self, nd_indexes: np.ndarray, nd_items: np.ndarray):
        """ params: nd_indexes: 1D int numpy array of indexes """
        """ param: nd_items: Corresponding numpy array of actions"""
        self.buffer.act[nd_indexes] = nd_items


class WorkerBuffer():

    def __init__(self, shared_mems: SharedMemoryGenerator.SharedRollouts):
        self.buffer = SharedMemoryGenerator.SharedRollouts(*buffers)

    def __getitem__(self, key: int) -> Union[np.ndarray, int]:
        return self.buffer.act[key]

    def __setitem__(self,
                    key: int,
                    item: Sequence[
                        np.ndarray,
                        Union[np.ndarray, float],
                        Union[np.ndarray, int, bool]
                    ]):
        self.buffer.obs[key] = item[0]
        self.buffer.reward[key] = item[1]
        self.buffer.done[key] = item[2]


class AsyncVecEnv(VecEnv):
    """
    """

    def __init__(self, env_fns, start_method=None,
                 n_env_per_core: int = 1, batchsize: int = None):
        self.waiting = False
        self.closed = False
        self.n_env_per_core = n_env_per_core
        n_envs = len(env_fns)

        if n_envs % n_env_per_core != 0:
            raise ValueError("Environments cannot be diveded equally")
        self.n_procs = n_envs / n_env_per_core

        if batchsize > n_envs:
            raise NotImplementedError(
                "Larger than #envs batchsizes are not supported")

        self.batchsize = batchsize
        if batchsize is None:
            self.batchsize = n_envs

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        # Shared memories
        self.memgen = SharedMemoryGenerator(env_fns[0], len(env_fns))
        shm = self.memgen.make_shared_mems()

        # Job queues
        self.job_queue = [mp.Queue(n_env_per_core) for i in range(n_procs)]
        self.ready_queue = mp.Queue(n_procs * n_env_per_core)

        # Queue maps
        # self.job_proc_map = {
        #     i: i // n_env_per_core
        #     for i in range(n_env_per_core * n_procs)
        # }
        # self.proc_job_map = {
        #     i: list(range(i * n_env_per_core, (i + 1) * n_env_per_core))
        #     for i in range(n_procs)
        # }

        self.processes = []
        for rank in range(n_procs):
            args = (
                ranks,
                CloudpickleWrapper(env_fn),
                shm,
                self.job_queue[rank],
                self.ready_queue
            )
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)

        VecEnv.__init__(self, n_envs, *self.memgen.env_spaces)

    def step_async(self, actions):
        # MODIFY ACTION SENDING / PUSH INDEX INTO JOB
        pass
        self.waiting = True

    def step_wait(self):
        # MODIFY STATE RECIEVE / GET INDEX FROM READY
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seed + idx))
        return [remote.recv() for remote in self.remotes]

    # def accumulate_batch(self):
    #     """ Accumulate batch of transitions in case of large batchsize """
    #     obs = np.empty((self.batchsize, *self.memgen.envinfo.obs.shape),
    #                    dtype=self.memgen.envinfo.obs.dtype)
    #     rewards = np.empty((self.batchsize, 1), dtype=np.float32)
    #     dones = np.empty((self.batchsize, 1), dtype=np.float32)

    def reset(self):
        for ix in self.num_envs:
            rank = ix // self.n_env_per_core
            self.job_queue[rank].push(ix)

        self.waiting = True
        batch_ix = [ready_queue.get() for i in range(self.batchsize)]
        self.waiting = False

        return self.memgen[np.array(batch_ix, dtype=np.int64)].obs
        # return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', 'rgb_array'))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)


if __name__ == "__main__":
    pass
