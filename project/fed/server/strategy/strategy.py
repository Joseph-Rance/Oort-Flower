"""A custom strategy for using traces."""

from typing import Any
from logging import INFO

import numpy as np

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from flwr.server.strategy import FedAvg

from project.client.client import IntentionalDropoutError
from project.fed.server.active_client_manager import ActiveClientManager


# flake8: noqa: E501
class FedAvgTraces(FedAvg):
    """Configurable FedAvg strategy implementation."""

    def __init__(
        self, *args: Any, **kwargs: Any
    ) -> None:
        self.current_virtual_clock = 0.
        super().__init__(*args, **kwargs)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:

        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        config["current_virtual_clock"] = self.current_virtual_clock

        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            server_round=server_round,
            current_virtual_clock=self.current_virtual_clock,
        )

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:

        for failure in failures:
            try:
                if isinstance(failure, BaseException):
                    raise failure
            except IntentionalDropoutError as e:
                log(INFO, f"IntentionalDropoutError: {e}")

        client_completion_times = [
            res.metrics["client_completion_time"] for _, res in results
        ]
        log(INFO, f"Completion times of clients: {client_completion_times}; clock: {self.current_virtual_clock}")
        self.current_virtual_clock += np.max(client_completion_times)

        return super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )

    # mostly the same as in FedAvg
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:

        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        metrics_aggregated["end_time"] = self.current_virtual_clock

        return loss_aggregated, metrics_aggregated
