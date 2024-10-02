from typing import Any


class Step:
    _state = {}

    def __init__(self, id: str) -> None:
        self.id = id

    def __call__(self, original_class) -> Any:
        def get_id(inner_self):
            return self.id

        def set_state(_, k, v):
            self._state[k] = v

        def get_state(_, k):
            return self._state[k]

        original_class.get_id = get_id
        original_class.get_state = get_state
        original_class.set_state = set_state

        return original_class
