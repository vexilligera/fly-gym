"""One worker owns CUDA/Brian2 and the optional visual MuJoCo environment."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from brain.model import ConnectomeBrain


class BrainService:
    def __init__(self, backend='brian2'):
        self.backend = backend
        self.progress = 'Starting full connectome backend'
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fly-brain')
        self.lock = Lock()
        self.navigation = None
        self.navigation_running = False
        self.navigation_generation = 0
        self.navigation_state = {'status': 'idle', 'message': 'Start a visual navigation trial'}
        self.future = self.pool.submit(self._load)

    def _load(self):
        def progress(message):
            self.progress = message
            print(f'Brain: {message}', flush=True)
        return ConnectomeBrain(progress, backend=self.backend)

    def status(self):
        if not self.future.done():
            return {'status': 'loading', 'message': self.progress}
        if self.future.exception():
            return {'status': 'error', 'message': str(self.future.exception())}
        return {'status': 'ready', 'metadata': self.future.result().metadata}

    def geometry(self):
        if not self.future.done():
            raise ValueError('Brain is still loading')
        return self.future.result().geometry

    def neuron_info(self, index):
        if not self.future.done():
            raise ValueError('Brain is still loading')
        return self.future.result().neuron_info(index)

    def execute(self, action, arguments):
        if not self.future.done():
            raise ValueError('Brain is still loading')
        if not self.lock.acquire(blocking=False):
            raise ValueError('Brain is busy with another simulation request')
        try:
            brain = self.future.result()
            future = self.pool.submit(self._manual, brain, action, arguments)
            return future.result(timeout=120)
        finally:
            self.lock.release()

    def _manual(self, brain, action, arguments):
        if self.navigation_running:
            raise ValueError('Pause the visual navigation trial before using manual stimulation')
        if self.navigation is not None:
            self.navigation.world.close()
            self.navigation = None
            self.navigation_state = {'status': 'idle', 'message': 'Manual stimulation owns the shared brain; start a new visual trial'}
        if action == 'reset':
            return brain.reset()
        if action == 'step':
            return brain.step(**arguments)
        raise ValueError('Unknown action')

    def vision_command(self, action, arguments):
        if self.status()['status'] != 'ready':
            raise ValueError('Brain is not ready')
        if not self.lock.acquire(blocking=False):
            raise ValueError('Another simulation command is in progress')
        try:
            return self.pool.submit(self._vision_command, action, arguments).result(timeout=120)
        finally:
            self.lock.release()

    def _vision_command(self, action, arguments):
        if action not in ('start', 'pause', 'reset'):
            raise ValueError('Unknown navigation action')
        if action == 'pause':
            self.navigation_running = False
            self.navigation_generation += 1
            self.navigation_state = {**self.navigation_state, 'status': 'paused'}
            return self.navigation_state
        if self.navigation is None:
            self.progress = 'Constructing the fly, compound eyes, and visual arena'
            from brain.navigation import VisualNavigation
            self.navigation = VisualNavigation(self.future.result())
        # Reset validates all arguments before modifying either model.
        if action == 'reset' or arguments or self.navigation.done:
            self.navigation_state = self.navigation.reset(**(arguments or self.navigation.config))
        self.navigation_running = action == 'start'
        self.navigation_generation += 1
        self.navigation_state = {**self.navigation.state, 'status': 'running' if self.navigation_running else 'paused'}
        if self.navigation_running:
            self.pool.submit(self._vision_tick, self.navigation_generation)
        return self.navigation_state

    def _vision_tick(self, generation):
        if not self.navigation_running or generation != self.navigation_generation:
            return
        try:
            self.navigation_state = self.navigation.step()
            if self.navigation.done:
                self.navigation_running = False
            else:
                # Queue one bin at a time so pause/reset commands can run
                # between bins on the SAME CUDA/OpenGL-owning worker.
                self.pool.submit(self._vision_tick, generation)
        except Exception as error:
            self.navigation_running = False
            self.navigation_state = {**self.navigation_state, 'status': 'error', 'message': str(error)}
            print(f'Visual navigation error: {error}', flush=True)
