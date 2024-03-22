"""The default client implementation.

Make sure the model and dataset are not loaded before the fit function.
"""

from copy import copy
from typing import Any
import random
from pathlib import Path
import pickle

import flwr as fl
from flwr.common import NDArrays
from pydantic import BaseModel
from torch import nn

from flwr.common.typing import Scalar

from project.fed.utils.utils import (
    generic_get_parameters,
    generic_set_parameters,
    get_isolated_rng_tuple,
)
from project.types.common import (
    CID,
    ClientDataloaderGen,
    ClientGen,
    EvalRes,
    FitRes,
    NetGen,
    TestFunc,
    TrainFunc
)
from project.utils.utils import obtain_device


class IntentionalDropoutError(BaseException): pass

def is_active(
    client_trace: dict[str, Any],
    current_time: int,
) -> bool:

    transformed_time = current_time % client_trace["finish_time"]

    # use weak inequality to include the round listed as the first
    active = [a for a in client_trace["active"] if a <= transformed_time]
    last_active = max(active) if active else -2

    inactive = [a for a in client_trace["inactive"] if a <= transformed_time]
    last_inactive = max(inactive) if inactive else -1

    return last_active > last_inactive

def get_client_completion_time(
    client_capacity: dict[str, float],
    computation_factor: float,
    communication_factor: float,
    n_data: int
) -> dict[str, float]:

    '''
    Computation is inference speed in ms/sample. A backward pass takes 2x the time as a forward pass.
    Communication is network speed in kB/s. We assume the model is 25MB (ResNet50)
    From this, we compute both times (multiplied by weighting parameters) in seconds
    '''

    return {
        "computation": client_capacity["computation"] * n_data * 3 / 1000 * computation_factor,
        "communication": 25_000 / client_capacity["communication"] * 2 * communication_factor
    }

class ClientConfig(BaseModel):
    """Fit/eval config, allows '.' member access and static checking.

    Used to check whether each component has its own independent config present. Each
    component should then use its own Pydantic model to validate its config. For
    anything extra, use the extra field as a simple dict.
    """

    # Instantiate model
    net_config: dict
    # Instantiate dataloader
    dataloader_config: dict
    # For train/test
    run_config: dict
    # Additional params used like a Dict
    extra: dict

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


class Client(fl.client.NumPyClient):
    """Virtual client for ray."""

    def __init__(
        self,
        cid: CID,
        working_dir: Path,
        net_generator: NetGen,
        dataloader_gen: ClientDataloaderGen,
        train: TrainFunc,
        test: TestFunc,
        client_seed: int,
        client_trace: dict[str, Any],
        client_capacity: dict[str, Any]
    ) -> None:
        """Initialize the client.

        Only ever instantiate the model or load dataset
        inside fit/eval, never in init.

        Parameters
        ----------
        cid : int | str | Path
            The client's ID.
        working_dir : Path
            The path to the working directory.
        net_generator : NetGen
            The network generator.
        dataloader_gen : ClientDataloaderGen
            The dataloader generator.
            Uses the client id to determine partition.

        Returns
        -------
        None
        """
        super().__init__()
        self.cid = cid
        self.net_generator = net_generator
        self.working_dir = working_dir
        self.net: nn.Module | None = None
        self.dataloader_gen = dataloader_gen
        self.train = train
        self.test = test

        self.client_capacity = client_capacity
        self.client_trace = client_trace

        # For deterministic client execution
        # The client_seed is generated from a specific Generator
        self.client_seed = client_seed
        self.rng_tuple = get_isolated_rng_tuple(self.client_seed, obtain_device())

    def fit(
        self,
        parameters: NDArrays,
        _config: dict,
    ) -> FitRes:
        """Fit the model using the provided parameters.

        Only ever instantiate the model or load dataset
        inside fit, never in init.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to use for training.
        _config : Dict
            The configuration for the training.
            Uses the pydantic model for static checking.

        Returns
        -------
        FitRes
            The parameters after training, the number of samples used and the metrics.
        """

        current_virtual_clock = _config["current_virtual_clock"]
        del _config["current_virtual_clock"]

        config: ClientConfig = ClientConfig(**_config)
        del _config

        config.run_config["device"] = obtain_device()

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )
        trainloader = self.dataloader_gen(
            self.cid,
            False,
            config.dataloader_config,
            self.rng_tuple,
        )
        num_samples, metrics = self.train(
            self.net,
            trainloader,
            config.run_config,
            self.working_dir,
            self.rng_tuple,
        )

        times = get_client_completion_time(
            client_capacity=self.client_capacity,
            computation_factor=1,
            communication_factor=1,
            n_data=num_samples * config.run_config["epochs"]
        )
        metrics["client_completion_time"] = times["communication"] + times["computation"]

        if not is_active(self.properties["traces"], int(current_virtual_clock + metrics["client_completion_time"])):
            raise IntentionalDropoutError(f"Client {self.cid} is no longer active")

        metrics["utility"] = num_samples * metrics["train_loss"]

        return (
            self.get_parameters({}),
            num_samples,
            metrics,
        )

    def _get_lr(self, lr, training_round) -> float:
        if training_round < 100:
            return lr
        if 49 < training_round < 150:
            return lr * 0.2
        if training_round < 180:
            return lr * 0.01
        return lr * 0.001

    def evaluate(
        self,
        parameters: NDArrays,
        _config: dict,
    ) -> EvalRes:
        """Evaluate the model using the provided parameters.

        Only ever instantiate the model or load dataset
        inside eval, never in init.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to use for evaluation.
        _config : Dict
            The configuration for the evaluation.
            Uses the pydantic model for static checking.

        Returns
        -------
        EvalRes
            The loss, the number of samples used and the metrics.
        """
        return 0., 1, {"accuracy": 0.}
        config: ClientConfig = ClientConfig(**_config)
        del _config

        config.run_config["device"] = obtain_device()

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )
        testloader = self.dataloader_gen(
            self.cid,
            True,
            config.dataloader_config,
            self.rng_tuple,
        )

        config.run_config = copy(config.run_config)  # avoid mutation

        # applies schedule to lr
        config.run_config["learning_rate"] = self._get_lr(config.run_config["learning_rate"], config.extra["server_round"])

        loss, num_samples, metrics = self.test(
            self.net,
            testloader,
            config.run_config,
            self.working_dir,
            self.rng_tuple,
        )
        return loss, num_samples, metrics

    def get_parameters(self, config: dict) -> NDArrays:
        """Obtain client parameters.

        If the network is currently none,generate a network using the net_generator.

        Parameters
        ----------
        config : Dict
            The configuration for the training.

        Returns
        -------
        NDArrays
            The parameters of the network.
        """
        if self.net is None:
            except_str: str = """Network is None.
                Call set_parameters first and
                do not use this template without a get_initial_parameters function.
            """
            raise ValueError(
                except_str,
            )

        return generic_get_parameters(self.net)

    def set_parameters(
        self,
        parameters: NDArrays,
        config: dict,
    ) -> nn.Module:
        """Set client parameters.

        First generated the network. Only call this in fit/eval.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to set.
        config : Dict
            The configuration for the network generator.

        Returns
        -------
        nn.Module
            The network with the new parameters.
        """
        net = self.net_generator(config, self.rng_tuple)
        generic_set_parameters(
            net,
            parameters,
            to_copy=False,
        )
        return net

    def __repr__(self) -> str:
        """Implement the string representation based on cid."""
        return f"Client(cid={self.cid})"

    def get_properties(self, config: dict) -> dict:
        """Implement how to get properties."""
        return {"traces": self.client_trace}


def get_client_generator(
    working_dir: Path,
    net_generator: NetGen,
    dataloader_gen: ClientDataloaderGen,
    train: TrainFunc,
    test: TestFunc,
    client_seed_generator: random.Random,
    num_clients: int
) -> ClientGen:
    """Return a function which creates a new Client.

    Client has access to the working dir,
    can generate a network and can generate a dataloader.
    The client receives train and test functions with pre-defined APIs.

    Parameters
    ----------
    working_dir : Path
        The path to the working directory.
    net_generator : NetGen
        The network generator.
        Please respect the pydantic schema.
    dataloader_gen : ClientDataloaderGen
        The dataloader generator.
        Uses the client id to determine partition.
        Please respect the pydantic schema.
    train : TrainFunc
        The train function.
        Please respect the interface and pydantic schema.
    test : TestFunc
        The test function.
        Please respect the interface and pydantic schema.
    seed : int
        The global seed for the random number generators.
    random_state : tuple[Any,Any,Any]
        The random state for the random number generator.
    np_random_state : dict[str,Any]
        The numpy random state for the random number generator.
    torch_random_state : torch.Tensor

    Returns
    -------
    ClientGen
        The function which creates a new Client.
    """

    with open("data/client_behave_trace.pkl", 'rb') as fin:
        client_traces = pickle.load(fin)

    with open("data/client_device_capacity.pkl", 'rb') as fin:
        client_capacities = pickle.load(fin)

    def client_generator(cid: CID) -> Client:
        """Return a new Client.

        Parameters
        ----------
        cid : int | str | Path
            The client's ID.

        Returns
        -------
        Client
            The new Client.
        """
        return Client(
            cid,
            working_dir,
            net_generator,
            dataloader_gen,
            train,
            test,
            client_seed=client_seed_generator.randint(0, 2**32 - 1),
            # for some reason there is no cid 0 in these so reduce cids by 1
            client_trace=client_traces[(int(cid)%len(client_capacities))+1],
            client_capacity=client_capacities[(int(cid)%len(client_capacities))+1],
        )

    return client_generator
