from project.object.task_handler import TaskHandler
from project.data.study.database import ColorDB

if __name__ == '__main__':
    task = TaskHandler()
    task.eval_speaker_with_listener(7, 9, test=True)

