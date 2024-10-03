import abc
from typing import Any, Dict


class BaseStep:
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "initialise")
            and callable(subclass.initialise)
            and hasattr(subclass, "run")
            and callable(subclass.run)
            and hasattr(subclass, "cleanup")
            and callable(subclass.cleanup)
            or NotImplemented
        )

    @abc.abstractmethod
    def initialise(self, infra: Dict[str, int], **kwargs) -> Any:
        """Initialise pipeline step"""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, **kwargs) -> Any:
        """Run pipeline step"""
        raise NotImplementedError

    @abc.abstractmethod
    def cleanup(self) -> Any:
        """Deallocate pipeline step resources"""
        raise NotImplementedError
