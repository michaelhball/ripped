from abc import ABCMeta, abstractmethod


class BaseWrapper(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass
    
    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def train():
        pass
