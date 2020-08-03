import multiprocessing as mp
import ctypes
from collections import OrderedDict, namedtuple
from typing import Union, Type, Optional, Dict, Any, List, Tuple, Callable, Sequence, NamedTuple
import gym
from gym.spaces import Box, Discrete
import numpy as np
import time

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper
from stable_baselines3.common.buffers import SharedRolloutBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape


def worker():
    """ Environment running function. Run multiple environment each with an
    associated index. Worker communicate with the master process via two
    queues namely, ready queue and job queue. When an index arrives into job
    queue, worker process run the environment with the index. When the
    environment is done stepping, worker process pushes the index into the
    ready queue. Environemnts are run in series within a worker process. In
    environment step, shared memory partition pointed by the index is first
    read to obtain action information and written back with the data that is
    returned by the environment.

        Worker process take commands from the shared queue. In the queue, we
    use three tuple that includes rank, index and the job information to
    communicate between the master process.
    """
    pass


class Timer():

    # TODO: Implement "with" statement

    def __init__(self):
        self.wait_time = 0
        self.process_time = 0
        self._waiting = False
        self.last_t = time.time()

    @property
    def waiting(self):
        return self._waiting

    @waiting.setter
    def waiting(self, value):
        current_t = time.time()
        if value is False and self._waiting is True:
            self.wait_time += current_t - self.last_t
            self.last_t = current_t
        elif value is True and self._waiting is False:
            self.process_time += current_t - self.last_t
        else:
            return None
        self.last_t = current_t
        self._waiting = value


class JobTuple(NamedTuple):
    index: Union[int, List[int], np.ndarray]
    info: Union[str, None]


class AsyncVecEnv(VecEnv):
    """
    """

    def __init__(self, env_fns, start_method=None,
                 n_env_per_core: int = 1,
                 forwardsize: int = None):
        n_envs = len(env_fns)
        self.timer = Timer()
        self.timer.waiting = False
        self.closed = False
        self.n_env_per_core = n_env_per_core
        self.worker_count = 0

        if n_envs % n_env_per_core != 0:
            raise ValueError("Environments cannot be diveded equally")
        self.n_procs = n_envs // n_env_per_core

        self.forwardsize = forwardsize
        if forwardsize is None:
            self.forwardsize = n_envs

        if self.forwardsize > n_envs:
            raise NotImplementedError(
                "Larger than #envs forwardsizes are not supported")

        # if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            # forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            # start_method = 'forkserver' if forkserver_available else 'spawn'
        # ctx = multiprocessing.get_context(start_method)
        # TODO: Default only for now

        env = env_fns[0]()
        obs_space = env.observation_space
        act_space = env.action_space
        del env

        # Shared memories
        self.sharedbuffer = SharedRolloutBuffer(
            observation_space=obs_space,
            action_space=act_space,
            n_envs=len(env_fns)
        )
        shm = self.sharedbuffer.shared_mem

        # Job queues
        self.job_queue = [mp.Queue(n_env_per_core) for i in range(self.n_procs)]
        self.ready_queue = mp.Queue(self.n_procs * n_env_per_core)

        self.processes = []
        for rank in range(self.n_procs):
            args = (
                rank,
                CloudpickleWrapper(
                    env_fns[
                        rank * self.n_env_per_core:
                        (rank + 1) * self.n_env_per_core
                    ]
                ),
                shm,
                self.job_queue[rank],
                self.ready_queue
            )
            # daemon=True: if the main process crashes, we should not cause
            # things to hang
            process = mp.Process(
                target=_worker,
                args=args,
                daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)

        super().__init__(n_envs, obs_space, act_space)

    def step(self, actions, jobtuple: JobTuple):
        """
        Step the environments with the given action

        :param actions: ([int] or [float]) the action
        :param jobtuple: job indexes of the given actions
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        self.sharedbuffer.buffer[jobtuple.index] = actions
        self.push_jobs(JobTuple(jobtuple.index, ["step"] * len(jobtuple.index)))

        self.timer.waiting = True
        indexes, infos = self.pull_ready_jobs(self.forwardsize)
        self.timer.waiting = False

        return (self.sharedbuffer.buffer.observations[indexes],
                self.sharedbuffer.buffer.rewards[indexes],
                self.sharedbuffer.buffer.dones[indexes],
                JobTuple(indexes, infos))

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seed + idx))
        return [remote.recv() for remote in self.remotes]

    def push_jobs(self, jobtuple: JobTuple):
        for ix, info in zip(jobtuple.index, jobtuple.info):
            rank = ix // self.n_env_per_core
            self.job_queue[rank].push(JobTuple(ix, info))
        self.worker_count += len(jobtuple.index)

    def pull_ready_jobs(self, size: int):
        """ Gather the jobTuples from the ready workers and distribute them to
        indexes and infos (job descriptions) """
        indexes, infos = list(zip(ready_queue.get() for i in range(size)))
        self.worker_count -= size
        return indexes, infos

    def reset(self):
        """
            Reset is only called at the very beginning of the training. When
        its called, master process assigns all the jobs to worker processes
        at once. Whenever the first batch of jobs are completed, job indexes
        and observations are returned so that the rollout can store them.
        """

        self.push_jobs(
            JobTuple(
                list(range(self.num_envs)),
                ["reset"] * self.num_envs
            )
        )
        self.timer.waiting = True
        indexes, infos = self.pull_ready_jobs(self.forwardsize)
        self.timer.waiting = False

        return (self.sharedbuffer.buffer.observations[indexes],
                JobTuple(indexes, infos))

    def close(self):
        if self.closed:
            return
        # Wait for all workers to finish their last job
        self.pull_ready_jobs(self.worker_count)

        # Send closing messages to workers
        self.push_jobs(
            JobTuple(
                list(range(self.num_envs)),
                ["close"] * self.num_envs
            )
        )

        # Join processes
        for process in self.processes:
            process.join()
        self.closed = True

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def step_async(self):
        raise NotImplementedError("Step is implmented")

    def step_wait(self):
        raise NotImplementedError("Step is implmented")

    # <--------------------------------- TODO ---------------------------------

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', 'rgb_array'))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

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

    # ---------------------------------- TODO -------------------------------->


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
