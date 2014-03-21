import luigi

class TaskPreloadingHint(object):

    def preload(self):
        pass

class PreloadingWorker(luigi.worker.Worker):

    def _fork_task(self, children, task_id):
        # TODO: This is nasty, but that is the only way to get the task from a child class
        task = self._Worker__scheduled_tasks[task_id]
        if isinstance(task, TaskPreloadingHint):
            task.preload()
        super(PreloadingWorker, self)._fork_task(children, task_id)

class PreloadingWorkerSchedulerFactory(luigi.interface.WorkerSchedulerFactory):
    def create_worker(self, scheduler, worker_processes):
        return PreloadingWorker(scheduler=scheduler, worker_processes=worker_processes)

def run(*args, **kwargs):
    return luigi.run(worker_scheduler_factory=PreloadingWorkerSchedulerFactory(), *args, **kwargs)
