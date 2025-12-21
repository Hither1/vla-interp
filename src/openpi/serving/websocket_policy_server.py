import asyncio
import http
import logging
import time
import traceback

import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


import os
import numpy as np

ACT_SAVE_DIR = "/n/netscratch/sham_lab/Lab/chloe00/pi0_activations"  # change path if you want
os.makedirs(ACT_SAVE_DIR, exist_ok=True)

# Episode bookkeeping (per connection)
EPISODE_COUNTER = 0  # counts connections

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        global EPISODE_COUNTER

        EPISODE_COUNTER += 1
        episode_id = EPISODE_COUNTER
        logger.info(f"Connection from {websocket.remote_address} opened (episode {episode_id})")

        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        current_episode_id = None
        episode_activations = []  # list of (num_layers, B, T, D), one per env step

        task_suite_name: str = (
            "libero_90"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
        )

        def flush_episode(episode_id, acts_list):
            """Save all timesteps of one episode into a single file."""
            if not acts_list or episode_id is None:
                return
            ep_array = np.stack(acts_list, axis=0)  # (time_steps, L, B, T, D)
            save_path = os.path.join(
                ACT_SAVE_DIR,
                f"{task_suite_name}_{episode_id}_post_ffn_last_step.npy",
            )
            np.save(save_path, ep_array)
            logger.info(
                "Saved activations for episode %s to %s (shape=%s)",
                episode_id,
                save_path,
                ep_array.shape,
            )

        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                episode_id = obs.get("episode_id", "unknown_episode")

                if episode_id != current_episode_id:
                    flush_episode(current_episode_id, episode_activations)
                    current_episode_id = episode_id
                    episode_activations = []

                from openpi.models import gemma
                gemma.ACTION_EXPERT_ACTS.clear()
                gemma._layer_counter.clear()
                

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time


                ###
                acts = gemma.ACTION_EXPERT_ACTS
                post_ffn_list = acts.get("block_post_ffn", [])

                if post_ffn_list:
                    
                    post_ffn_array = np.stack(
                        [np.asarray(x, dtype=np.float32) for x in post_ffn_list],
                        axis=0,
                    )
                    print(post_ffn_array)

                    NUM_LAYERS = 18  # e.g., gemma_300m / gemma_f needed

                    assert post_ffn_array.shape[0] % NUM_LAYERS == 0, (
                        "Mismatch between number of recorded activations "
                        "and assumed number of layers"
                    )
                    num_diff_steps = post_ffn_array.shape[0] // NUM_LAYERS
                    B, D = post_ffn_array.shape[1:]
                    post_ffn_array = post_ffn_array.reshape(
                        num_diff_steps, NUM_LAYERS, B, D
                    )

                    # Keep only the **last diffusion step**: shape (num_layers, B, T, D)
                    last_step_acts = post_ffn_array[-1]  # (L, B, T, D)

                    # Append to episode buffer: time_step axis grows here
                    episode_activations.append(last_step_acts)

                    logger.info(
                        "Recorded activations for episode %s, timestep %d "
                        "(last diffusion step, %d layers, B=%d, D=%d)",
                        episode_id,
                        len(episode_activations) - 1,
                        NUM_LAYERS,
                        B,
                        D,
                    )
                ###

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

        flush_episode(current_episode_id, episode_activations)

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
