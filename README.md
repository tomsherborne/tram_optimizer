# TRAM: Bridging Trust Regions and Sharpness Aware Minimization
### Tom Sherborne, Naomi Saphra, Pradeep Dasigi, Hao Peng

Code for [TRAM: Bridging Trust Regions and Sharpness Aware Minimization](https://arxiv.org/abs/2310.03646) submitted to ICLR 2024.

_I am working on my PhD write up at present but I plan to fully expand on this repo when I have the bandwidth_

The minimal requirements are `tram.py`, `parameter_target.py` and a training step that looks something like the snippet below. This is meant to replace the `training_step` function of the Huggingface `Trainer` class. Our version of this class will be added soon.

The critical details are:

    - Define a parameter target object which specifies the other model to compare the current model for trust region measurement.

    - Overload the optimizer with the `TRAM` class. `TRAM` accepts a base optimizer (e.g., Adam) and runs the TRAM logic on top. Remember to adjust the LR scheduler call to attach to `optimizer.base_optimizer`.

    - Define a `logit_distance` class which gives you back the KL divergence. See `kl.py`.



#### Example training step 

```python
def _training_step_tram(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
) -> torch.Tensor:
    """
    Perform a TRAM step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (`nn.Module`):
            The model to train.
        inputs (`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.

    Return:
        `torch.Tensor`: The tensor with training loss on this batch.
    """    
    def step(subsample_batch: bool = False, return_outputs: bool = False) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        with self.compute_loss_context_manager():
            outputs = None
            loss_outputs = self.compute_loss(model, inputs, return_outputs=return_outputs, subsample_batch=subsample_batch)

            if return_outputs:
                loss, outputs = loss_outputs
            else:
                loss = loss_outputs

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return (loss, outputs) if return_outputs else loss

    # Set to training mode
    model.train()

    # Enable batch norm updates on first pass
    enable_running_stats(model)

    inputs = self._prepare_inputs(inputs)

    # First forward pass
    loss, outputs = step(subsample_batch=False, return_outputs=True)
    
    # TRAM ascent includes noising fn
    logits = outputs.get("logits")
    target_logits = self.parameter_target.forward(self.model, inputs, noise_gradient=False).get("logits")
    logit_noise_divergence = self.logit_distance.get_divergence(target_logits, logits)
    
    logit_noise_divergence *= self.logit_scale # Scaling parameter which may be needed for CLM
    self.optimizer.first_step(logit_noise_divergence, zero_grad=True)

    # Don't update batch norm on second pass
    disable_running_stats(model)

    # Second Forward pass (includes .backward())
    loss = step()

    # The loss call. Second step() happens in Trainer._inner_training_loop
    return loss.detach() / self.args.gradient_accumulation_steps

```

#### Known Issues
    - There is a current issue with the `accelerate` codebase in HF. When functionality like adjusting the batch size gets called, the optimizer can break. 
    - Multi GPU training is not 100% verified. At present, this includes ignoring m-sharpness.
