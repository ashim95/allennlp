from typing import Tuple, Union, Optional, TYPE_CHECKING

from fairscale.nn import FullyShardedDataParallel as _FSDP
from fairscale.nn.wrap import enable_wrap, wrap
from fairscale.nn.misc import FlattenParamsWrapper
from fairscale.optim.grad_scaler import GradScaler
from overrides import overrides
import torch
from torch.cuda import amp

from allennlp.nn.parallel.sharded_module_mixin import ShardedModuleMixin
from allennlp.nn.parallel.ddp_wrapper import (
    DdpWrapper,
    DdpWrappedModel,
    StateDictType,
    LoadStateDictReturnType,
)

if TYPE_CHECKING:
    # To prevent circular imports
    from allennlp.models import Model


class FSDP(_FSDP, ShardedModuleMixin):
    @overrides
    def get_original_module(self) -> torch.nn.Module:
        module = self.module
        if isinstance(module, FlattenParamsWrapper):
            module = module.module
        return module


class FairScaleFsdpWrappedModel(DdpWrappedModel):
    @overrides
    def load_local_state_dict(
        self, state_dict: StateDictType, strict: bool = True
    ) -> LoadStateDictReturnType:
        return self.model.load_local_state_dict(state_dict, strict=strict)  # type: ignore[operator]

    @overrides
    def local_state_dict(self, *args, **kwargs) -> Optional[StateDictType]:
        return self.model.local_state_dict(*args, **kwargs)  # type: ignore[operator]

    @overrides
    def clip_grad_norm_(self, max_norm: Union[float, int]) -> torch.Tensor:
        return self.model.clip_grad_norm_(max_norm)  # type: ignore[operator]

    @overrides
    def get_grad_scaler(self) -> amp.GradScaler:
        return GradScaler()


@DdpWrapper.register("fairscale_fsdp")
class FairScaleFsdpWrapper(DdpWrapper):
    """
    A `DdpWrapper` for FairScale's [`FullyShardedDataParallel`]
    (https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html).
    """

    def __init__(
        self,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        cuda_device: Union[torch.device, int] = -1,
        mixed_precision: bool = False,
        reshard_after_forward: bool = True,
        flatten_parameters: bool = True,
    ) -> None:
        super().__init__(local_rank=local_rank, world_size=world_size, cuda_device=cuda_device)
        self._fsdp_kwargs = {
            "compute_device": self.cuda_device,
            "mixed_precision": mixed_precision,
            "reshard_after_forward": reshard_after_forward,
            "flatten_parameters": flatten_parameters,
        }
        if mixed_precision:
            self._fsdp_kwargs["move_params_to_cpu"] = True
            self._fsdp_kwargs["clear_autocast_cache"] = True

    @overrides
    def wrap_model(self, model: "Model") -> Tuple["Model", DdpWrappedModel]:
        wrapped_model = FSDP(
            model,
            **self._fsdp_kwargs,
        )
        if not self._fsdp_kwargs["mixed_precision"] and self.cuda_device != torch.device("cpu"):
            wrapped_model = wrapped_model.cuda()
        # `FSDP._lazy_init()` may have been called already on submodules that were wrapped
        # (through `wrap_module()`), leading those submodules to think they are root submodules.
        # So we need to call `FSDP._reset_lazy_init()` on any of these now.
        for module in wrapped_model.modules():
            if isinstance(module, FSDP):
                module._reset_lazy_init()
        return model, FairScaleFsdpWrappedModel(
            wrapped_model,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )

    @overrides
    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        with enable_wrap(wrapper_cls=FSDP, **self._fsdp_kwargs):
            wrapped_module = wrap(module)
        return wrapped_module