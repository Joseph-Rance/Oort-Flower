"""Implementation of Oort sampling as a client manager"""

import logging
from typing import Any
import math
import random
import numpy as np

from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from project.client.client import is_active

from flwr.common import (
    GetPropertiesIns,
    GetPropertiesRes
)


class OortClientManager(SimpleClientManager):

    def __init__(
        self,
        seed: int = 0,
        blacklist_rounds: int = 500,
        max_blacklist_length: int = 5,
        pacer_interval: float = 2,
        pacer_delta: float = 0.05,           # in interval [0, 1)
        utility_clip_prop: float = 0.95,     # in interval [0, 1)
        alpha: float = 2,
        epsilon: float = 0.9,                # in interval [0, 1]
        epsilon_decay: float = 0.98,         # in interval [0, 1]
        min_epsilon: float = 0.2,            # in interval [0, 1]
        utility_cutoff_pos: float = 0.95,    # in interval [0, 1)
        utility_cutoff_prop: float = 0.95    # in interval [0, 1)
    ) -> None:

        super().__init__()
        self.seed = seed

        self.blacklist_rounds = blacklist_rounds
        self.blacklist = []
        self.max_blacklist_length = max_blacklist_length

        self.pacer_interval = pacer_interval

        self.prev_pacer_util = 0
        self.curr_pacer_util = 0
        self.pacer_delta = pacer_delta
        self.train_time_cutoff = pacer_delta  # bigger is looser

        self.utility_clip_prop = utility_clip_prop

        self.alpha = alpha

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.utility_cutoff_pos = utility_cutoff_pos
        self.utility_cutoff_prop = utility_cutoff_prop

    def run_pacer(self, server_round: int) -> None:
        # NOTE: This pacer does not follow Alg 1 in the Oort paper
        # Instead, it follows the implementation provided by the authors on GitHub as this is more
        # thorough. The differences are minor

        if server_round == self.pacer_interval:
            self.prev_pacer_util = self.curr_pacer_util
            self.curr_pacer_util = 0

        if server_round >= 0 and server_round % self.pacer_interval == 0:

            if abs(self.curr_pacer_util - self.prev_pacer_util) <= self.prev_pacer_util * 0.1:
                self.train_time_cutoff = min(1, self.train_time_cutoff + self.pacer_delta)

            elif abs(self.curr_pacer_util - self.prev_pacer_util) >= self.prev_pacer_util * 5:
                self.train_time_cutoff = max(self.pacer_delta,
                                             self.train_time_cutoff - self.pacer_delta)

            self.prev_pacer_util = self.curr_pacer_util
            self.curr_pacer_util = 0

    def sample(
        self,
        num_clients: int,
        min_num_clients: int | None = None,
        server_round: int | None = None,
        current_virtual_clock: float | None = None,
        properties: dict[str, Any] | None = None
    ) -> list[ClientProxy]:

        if min_num_clients is None:
            min_num_clients = num_clients

        if server_round is None:
            server_round = 0

        if current_virtual_clock is None:
            current_virtual_clock = 0

        if num_clients == 0:
            return []


        # wait for clients to be available
        self.wait_for(min_num_clients)

        # shuffle client order (only necessary for early rounds, but run for all just in case)
        cids = list(self.clients)
        random.seed(self.seed)
        for _ in range(server_round):
            random.shuffle(cids)

        # get list of which clients are active *IN THE SIMULATUION*
        available_clients = []

        ins = GetPropertiesIns(config={
            "traces": "Dict[str, Any]"
        })

        for cid in cids:
            value = self.clients[cid].get_properties(ins, timeout=None)
            if is_active(value.properties["traces"], current_virtual_clock):
                available_clients.append(cid)

        log(logging.INFO, "Available clients: %s", available_clients)

        # on first round, return random clients (would be same if ran full Oort)
        if server_round <= 1:
            # We may want to put a warning here if we do not return min_num_clients
            # It is omitted because it is expected to be thrown regularly
            return [self.clients[cid] for cid in available_clients[:num_clients]]

        # get list of which clients can be selected *IN THE SIMULATUION* (active & not blacklisted)
        selectable_clients = []

        for cid in available_clients:
            if properties[cid]["rounds"] >= self.blacklist_rounds \
                    and len(self.blacklist) < self.max_blacklist_length:
                self.blacklist.append(cid)
            else:
                selectable_clients.append(cid)

        # clip utility values
        utility_bound_idx = int(len(selectable_clients) * self.utility_clip_prop)
        utility_clip_bound = np.partition([
            properties[cid]["utility"] for cid in selectable_clients
        ], utility_bound_idx)[utility_bound_idx]
        for cid in selectable_clients:
            properties[cid]["utility"] = max(properties[cid]["utility"], utility_clip_bound)

        self.run_pacer(server_round)  # updates self.train_time_cutoff

        # compute preferred train time
        # NOTE: this differs from Alg 1 in paper. Instead, I have followed the slighly modified
        # implementation provided by the authors on GitHub
        times = [
            properties[cid]["time"] for cid in selectable_clients
                if properties[cid]["last_sampled"] is not None
        ]

        cutoff_time = min(int(len(times) * self.train_time_cutoff), len(times)-1)
        target_train_time = np.partition(times, cutoff_time)[cutoff_time]

        utilities = [properties[cid]["utility"] for cid in selectable_clients]
        max_utility, min_utility = max(utilities), min(utilities)

        # update client to include the temporal uncertainty and global system utility terms
        utilities = []
        for cid in selectable_clients:
            if properties[cid]["last_sampled"] is None:
                utilities.append(-float("inf"))
            else:
                utilities.append(
                    (properties[cid]["utility"] - min_utility) / max(1e-5, max_utility - min_utility) \
                * max(1, (target_train_time/max(1e-5, properties[cid]["time"])) ** self.alpha) \
                + math.sqrt(0.1*math.log(current_virtual_clock)/max(1e-5, properties[cid]["last_sampled"]))
                )

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

        # select clients with c% of cutoff utility, then sample by utility
        idx = int(len(utilities) * self.utility_cutoff_pos)
        cutoff_utility = np.partition(utilities, idx)[idx] * self.utility_cutoff_prop

        higher_clients = [
            cid for i, cid in enumerate(selectable_clients) if utilities[i] >= cutoff_utility
        ]
        higher_utilities = [properties[cid]["utility"] for cid in higher_clients]
        selected_exploitation_clients = np.random.choice(
            higher_clients,
            min(len(higher_clients), int(num_clients*(1-self.epsilon))),
            p=[u/sum(higher_utilities) for u in higher_utilities],
            replace=False
        ).tolist()

        expl_utilities = [properties[cid]["utility"] for cid in higher_clients]
        self.curr_pacer_util += sum(expl_utilities)/len(expl_utilities)

        # sample unexplored clients by speed
        remaining_clients = [
            cid for cid in selectable_clients if cid not in selected_exploitation_clients
        ]
        remaining_speeds = [
            max(1, (target_train_time/max(1e-5, properties[cid]["time"])) ** self.alpha) for cid in remaining_clients
        ]

        selected_exploration_clients = np.random.choice(
            remaining_clients,
            min(len(remaining_clients), num_clients-len(selected_exploitation_clients)),
            p=[v/sum(remaining_speeds) for v in remaining_speeds],
            replace=False
        ).tolist()

        # combine exploration and exploitation samples
        selected_clients = selected_exploitation_clients + selected_exploration_clients
        client_list = [self.clients[cid] for cid in selected_clients]

        log(logging.INFO, "Sampled the following clients: %s", selected_clients)
        return client_list
