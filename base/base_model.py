import abc
from typing import Any


class BaseModel:
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "transcribe")
            and callable(subclass.transcribe)
            or NotImplemented
        )

    @abc.abstractmethod
    def transcribe(self, **kwargs) -> Any:
        """Run ASR model inference"""
        raise NotImplementedError
