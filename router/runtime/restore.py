from __future__ import annotations

from typing import Optional

from ..routing import RegimeComposer
from ..state import RouterState, router_state_from_jsonable


def restore_router_state(payload: object, *, composer: RegimeComposer) -> Optional[RouterState]:
    return router_state_from_jsonable(payload, composer.compose)
