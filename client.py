# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LifeOps Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LifeopsAction, LifeopsObservation


class LifeopsEnv(
    EnvClient[LifeopsAction, LifeopsObservation, State]
):
    """
    Client for the LifeOps Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    """

    def _step_payload(self, action: LifeopsAction) -> Dict:
        """
        Convert LifeopsAction to JSON payload for step message.
        """
        return action.dict()

    def _parse_result(self, payload: Dict) -> StepResult[LifeopsObservation]:
        """
        Parse server response into StepResult[LifeopsObservation].
        """
        obs_data = payload.get("observation", {})
        # Ensure all fields are present
        observation = LifeopsObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
