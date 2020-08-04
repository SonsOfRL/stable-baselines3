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
from stable_baselines3.common.buffers import MultiSharedRolloutBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape


def worker(rank,
           env_fns,
           shm,
           job_queue,
           ready_queue,
           buffer_size,
           n_envs,
           n_env_per_core):
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

    timer = Timer()
    envs = [env_fn() for env_fn in env_fns.var]
    shared_buffer = MultiSharedRolloutBuffer(
        buffer_size,
        envs[0].observation_space,
        envs[0].action_space,
        n_envs,
        shm
    )
    active_envs = len(envs)

    while active_envs > 0:
        timer.waiting = True
        jobtuple = job_queue.get()
        timer.waiting = False

        ix = jobtuple.index % n_env_per_core
        if jobtuple.info == "reset":
            state = envs[ix].reset()
            shared_buffer.add_last_obs(state, jobtuple.index)
            shared_buffer.add_obs(state, 0, jobtuple.index)
            ready_queue.put(JobTuple(
                jobtuple.index, 0, "act"))

        elif jobtuple.info == "step":
            act = shared_buffer.get_act(jobtuple.poses, jobtuple.index)
            if np.product(act.shape) == 1:
                act = act.astype(envs[ix].action_space.dtype)
                act = act.item()

            new_obs, reward, done, _ = envs[ix].step(act)
            shared_buffer.add_step(new_obs, float(reward), float(done),
                                   jobtuple.poses, jobtuple.index)
            ready_queue.put(JobTuple(
                jobtuple.index, jobtuple.poses + 1, "act"))

        elif jobtuple.info == "close":
            envs[ix].close()
            envs[ix] = None
            active_envs -= 1
            ready_queue.put(JobTuple(
                jobtuple.index, None, "closed"))


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
    poses: Union[int, List[int], np.ndarray, None]
    info: Union[str, List[str], None]


class AsyncVecEnv(VecEnv):
    """
    """

    def __init__(self, env_fns,
                 buffer_size: int,
                 start_method=None,
                 n_env_per_core: int = 1,
                 batchsize: int = None):
        n_envs = len(env_fns)
        self.timer = Timer()
        self.timer.waiting = False
        self.closed = False
        self.n_env_per_core = n_env_per_core
        self.worker_count = 0

        if n_envs % n_env_per_core != 0:
            raise ValueError("Environments cannot be equally partitioned")
        self.n_procs = n_envs // n_env_per_core

        self.batchsize = batchsize
        if batchsize is None:
            self.batchsize = n_envs

        if self.batchsize > n_envs:
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
        self.sharedbuffer = MultiSharedRolloutBuffer(
            buffer_size,
            observation_space=obs_space,
            action_space=act_space,
            n_envs=len(env_fns)
        )
        shm = self.sharedbuffer.shared_mem

        # Job queues
        self.job_queue = [mp.Queue(n_env_per_core) for i in range(self.n_procs)]
        self.ready_queue = mp.Queue(n_envs)

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
                self.ready_queue,
                buffer_size,
                n_envs,
                self.n_env_per_core
            )
            # daemon=True: if the main process crashes, we should not cause
            # things to hang
            process = mp.Process(
                target=worker,
                args=args,
                daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)

        super().__init__(n_envs, obs_space, act_space)

    def step(self, *args, **kwargs):
        """
        """
        raise RuntimeError("step method should not be called")

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seed + idx))
        return [remote.recv() for remote in self.remotes]

    def push_jobs(self, jobtuples: JobTuple):
        for ix, pos, info in zip(*jobtuples):
            rank = ix // self.n_env_per_core
            self.job_queue[rank].put(JobTuple(ix, pos, info))
        self.worker_count += len(jobtuples.index)

    def pull_ready_jobs(self) -> JobTuple:
        """ Gather the jobTuple(s) from the ready workers """
        self.worker_count -= 1
        return self.ready_queue.get()

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
                [None] * self.num_envs,
                ["reset"] * self.num_envs
            )
        )
        self.timer.waiting = True
        jobs = JobTuple(*zip(
            *(self.pull_ready_jobs() for _ in range(self.batchsize)))
        )
        self.timer.waiting = False

        return jobs

    def close(self):
        if self.closed:
            return
        # Wait for all workers to finish their last job
        for _ in range(self.worker_count):
            self.pull_ready_jobs()

        # Send closing messages to workers
        self.push_jobs(
            JobTuple(
                list(range(self.num_envs)),
                [None] * self.num_envs,
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
        raise NotImplementedError("Step is not implemented")

    def step_wait(self):
        raise NotImplementedError("Step is not implemented")

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
