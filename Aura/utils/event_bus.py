import sys

class ProgressObserver:
    def update(self, fraction: float, message: str):
        pass

class StderrProgressObserver(ProgressObserver):
    def update(self, fraction: float, message: str):
        sys.stderr.write(f"PROGRESS: {fraction:.2f}:{message}\n")
        sys.stderr.flush()

class ProgressSubject:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Prevent re-initialization if called via get_instance multiple times
        if not hasattr(self, 'observers'):
            self.observers = []

    def attach(self, observer: ProgressObserver):
        if observer not in self.observers:
            self.observers.append(observer)

    def detach(self, observer: ProgressObserver):
        if observer in self.observers:
            self.observers.remove(observer)

    def notify(self, fraction: float, message: str):
        for observer in self.observers:
            observer.update(fraction, message)

# Helper function to easily publish progress anywhere in the codebase
def publish_progress(fraction: float, message: str):
    ProgressSubject.get_instance().notify(fraction, message)
