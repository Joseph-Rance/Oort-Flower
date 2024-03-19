"""A client manager that guarantees deterministic client sampling."""

from typing import Any
import logging
import random

from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from project.client.client import is_active

from flwr.common import (
    GetPropertiesIns,
    GetPropertiesRes
)


class ActiveClientManager(SimpleClientManager):

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed

    def sample(
        self,
        num_clients: int,
        min_num_clients: int | None = None,
        server_round: int | None = None,
        current_virtual_clock: float | None = None,
        **kwargs: Any
    ) -> list[ClientProxy]:

        if min_num_clients is None:
            min_num_clients = num_clients

        if server_round is None:
            server_round = 0

        if current_virtual_clock is None:
            current_virtual_clock = 0

        # wait for clients to be available
        self.wait_for(min_num_clients)

        cids = list(self.clients)
        random.seed(self.seed)
        for _ in range(server_round):
            random.shuffle(cids)

        # get list of which clients are available *IN THE SIMULATUION*
        available_cids = []

        ins: GetPropertiesIns = GetPropertiesIns(config={
            "traces": "Dict[str, Any]"
        })

        while len(available_cids) < num_clients and len(cids) > 0:
            cid = cids.pop()
            value: GetPropertiesRes = self.clients[cid].get_properties(ins, timeout=None)
            if is_active(value.properties["traces"], current_virtual_clock):
                available_cids.append(cid)

        client_list = [self.clients[cid] for cid in available_cids]

        log(logging.INFO, "Sampled the following clients: %s", available_cids)
        return client_list
