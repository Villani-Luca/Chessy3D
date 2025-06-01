import sys
import traceback
from abc import ABC, abstractmethod, ABCMeta
from typing import Callable, TypeVar

from PySide6.QtCore import QObject, Signal, QRunnable, Slot


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:
        finished: no data
        error: tuple(exctype, value , traceback.format_exc() )
        result: object data returned from processing
        progress: int indicating % progress
    '''

    start = Signal()
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(float, str)

TSignal = TypeVar('TSignal')
class Worker[T: WorkerSignals](QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, signals: T = WorkerSignals()):
        super().__init__()

        # Store constructor arguments (re-used for processing)
        self.signals = signals

    def execute(self):
        pass

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            self.signals.start.emit()
            result = self.execute()
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class FunctionWorker(Worker):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''
    def __init__(self, fn: Callable, signals: WorkerSignals = WorkerSignals(), *args, **kwargs):
        super().__init__(signals)

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    def execute(self):
        return  self.fn(*self.args, **self.kwargs)
