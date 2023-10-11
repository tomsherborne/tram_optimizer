import logging
from abc import ABC, abstractmethod
from typing import Dict, Union, Any, List, Optional
from copy import deepcopy
import contextlib

from overrides import override
import torch
import torch.nn as nn
from torch import distributions
from torch import Tensor

import transformers
from transformers import PreTrainedModel
from torch.nn import DataParallel
from transformers import XLMRobertaForSequenceClassification, XLMRobertaForQuestionAnswering, GPT2LMHeadModel

logger = logging.getLogger(__name__)


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
        self.has_updated = False

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class ParameterTarget(ABC):
    """
    Provides an interface to return non-standard comparisons to existing model outputs.

    e.g.,
        - InputPerturbation returns the output of a forward pass with noise injected to the inputs
        - ParameterHistory returns the forward pass of inputs through a moving-average of weight history

    Class has three functions:
        1. __init__() sets up features such as EMA decay or noise variance/type
        2. update() updates the noise function with new information (can be a no-op)
        3. forward() receives a model and inputs and returns a SequenceClassifierOutput
                     of the model.forward() with relevant modifications
    """
    def __init__(self, *args, **kwargs):
        # pop specific keyword args and set them
        for arg in kwargs.copy():
            if arg in kwargs:
                val = kwargs.pop(arg)
                setattr(self, arg, val)
    
    @abstractmethod
    def init(self, model) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, model_state) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, model: transformers.PreTrainedModel, inputs: Dict[str, Union[Tensor, Any]], noise_gradient: bool = False) -> None:
        raise NotImplementedError


class InputPerturbationTarget(ParameterTarget):
    """
    Forward pass of f(x+z) where f is given model and z is sampled noise from self.noise_sampler
    """
    def __init__(self, *args, **kwargs):
        super(InputPerturbationTarget, self).__init__(*args, **kwargs)

        # Declare noise function
        self.noise_sampler = (
                distributions.normal.Normal(loc=0.0, scale=self.eps)
                if self.noise_type == "normal"
                else distributions.uniform.Uniform(low=-self.eps, high=self.eps)
            )
        
        self.embedding_fn = None

    @override
    def init(self, model) -> None:
        pass

    def update(self, model_state) -> None:
        # No - op function
        pass

    def model_class_to_embedding_function(self, model: PreTrainedModel) -> torch.nn.Module:
        # For the combined case we need to search the submodel
        if isinstance(model, ModelEmaV2) or isinstance(model, DataParallel): 
            if isinstance(model.module, XLMRobertaForSequenceClassification) or isinstance(model.module, XLMRobertaForQuestionAnswering):
                self.embedding_fn = lambda model: model.module.roberta.embeddings.word_embeddings
            elif isinstance(model.module, GPT2LMHeadModel):
                self.embedding_fn = lambda model: model.module.transformer.wte
        else:
            if isinstance(model, XLMRobertaForSequenceClassification) or isinstance(model, XLMRobertaForQuestionAnswering):
                self.embedding_fn = lambda model: model.roberta.embeddings.word_embeddings
            elif isinstance(model, GPT2LMHeadModel):
                self.embedding_fn = lambda model: model.transformer.wte
            else:
                raise NotImplementedError(f"InputPerturbationTarget is not implemented for class {type(model)}")
        
    @override
    def forward(self, model: PreTrainedModel, inputs: Dict[str, Union[Tensor, Any]], noise_gradient: bool = False) -> None:
        assert "input_ids" in inputs, f"inputs to InputPerturbationTarget require 'input_ids' key"
        input_ids = inputs.pop("input_ids")

        if self.embedding_fn is None:
            self.model_class_to_embedding_function(model)

        embedder = self.embedding_fn(model)
        model_embeddings = embedder(input_ids)

        if not noise_gradient:
            model_embeddings = model_embeddings.detach().clone()

        noise = self.noise_sampler.sample(sample_shape=model_embeddings.shape).to(model_embeddings)

        inputs['inputs_embeds'] = model_embeddings + noise

        ctx = contextlib.nullcontext if noise_gradient else torch.no_grad
        torch.cuda.empty_cache()

        with ctx():
            output = model(**inputs)

        return output
    
class ParameterHistoryTarget(ParameterTarget):
    """
    Forward pass of f(x) where f is some historical model.
    Historical updates of f occur after each step()
    if self.decay = 0, f is the model from the previous step
    if self.decay = 1, f is the pretrained model only
    if self.decay in (0,1), f is the EMA of models from the training history
    """
    def __init__(self, *args, **kwargs):
        super(ParameterHistoryTarget, self).__init__(*args, **kwargs)

        # Setup EMA model
        self.ema_model = None

    @override
    def init(self, model) -> None:
        if hasattr(model, "module"):
            model = model.module
        self.ema_model = ModelEmaV2(model=model, decay=self.decay, device=model.device)
        logger.info("EMA model setup!")
        
    @override
    def update(self, model_state) -> None:
        self.ema_model.update(model_state)

    @override
    def forward(self, model: PreTrainedModel, inputs: Dict[str, Union[Tensor, Any]], noise_gradient: bool = False) -> None:
        ctx = contextlib.nullcontext if noise_gradient else torch.no_grad
        torch.cuda.empty_cache()
        with ctx():
            output = self.ema_model.module(**inputs)
            
        return output
