"""All-block LoRA extension of the audited exact-boundary Ouro forward pass."""
from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
import re
import torch
from torch import nn

from experiments.loop_monitoring.model import LoopPolicy, ModelSpec
from .lora import attach, clone_with_shared_frozen_parameters, disabled, frozen_digest, adapter_state, load_adapter_state


class OuroLoRAPolicy(LoopPolicy):
    def __init__(self, base, spec, rank=64, alpha=128):
        nn.Module.__init__(self)
        if spec.scope != 'all' or spec.architecture not in ('tied', 'untied'):
            raise ValueError('Replication adaptation requires all-block tied/untied LoRA')
        self.base = base.requires_grad_(False)
        self.spec, self.config = spec, base.config
        self.rank, self.alpha = rank, alpha
        self._shared_initializer_state = None
        depth = len(base.model.layers)
        self.selected = [i for i in range(depth) if not (spec.freeze_last and i == depth - 1)]
        self.independent = nn.ModuleDict()
        self.targets = {}
        for i in self.selected:
            block = base.model.layers[i]
            # The same initial A matrices across passes isolate weight tying:
            # all B matrices are zero and initial policy functions are equal.
            rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            if spec.architecture == 'tied':
                self.targets[f'shared.{i}'] = attach(block, rank, alpha)
            else:
                for loop in range(spec.loops):
                    torch.random.set_rng_state(rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state_all(cuda_rng)
                    if str(loop) not in self.independent:
                        self.independent[str(loop)] = nn.ModuleDict()
                    cloned = clone_with_shared_frozen_parameters(block)
                    self.targets[f'{loop}.{i}'] = attach(cloned, rank, alpha)
                    self.independent[str(loop)][str(i)] = cloned
        self._frozen_digest = frozen_digest(self)

    @classmethod
    def load(cls, path, spec, initial_checkpoint=None, rank=64, alpha=128, dtype=torch.bfloat16):
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('Only explicitly audited BF16 or FP32 arithmetic is supported')
        if initial_checkpoint is not None:
            raise ValueError('Published-task adaptation starts from pretrained Ouro; old cheating SFT is a separate experiment')
        from experiments.loop_monitoring.vendor.ouro.modeling_ouro import OuroForCausalLM
        from experiments.loop_monitoring.vendor.ouro.configuration_ouro import OuroConfig
        cfg = OuroConfig.from_pretrained(path, local_files_only=True)
        base = OuroForCausalLM.from_pretrained(path, config=cfg, torch_dtype=dtype,
            attn_implementation='sdpa', local_files_only=True).cuda()
        # Move before copying module shells; frozen tensors then share GPU storage.
        return cls(base, spec, rank, alpha).eval()

    def reference(self):
        if self._shared_initializer_state is not None:
            return self.initializer_reference()
        return disabled(self)

    def initialize_shared(self, state):
        """Copy a tied/frozen initializer into every corresponding adapter pass.

        Extra last-block adapters stay zero-delta. This preserves the initializer
        function without merging/re-rounding weights. A real-model four-cell
        parity audit is still required before using it in the comparison.
        """
        if self._shared_initializer_state is not None:
            raise ValueError('Initializer already installed')
        depth = len(self.base.model.layers)
        current = adapter_state(self)
        used = set()
        for target in current:
            source = re.sub(r'^independent\.\d+\.(\d+)\.', r'base.model.layers.\1.', target)
            match = re.match(r'^base\.model\.layers\.(\d+)\.', source)
            if match is None:
                raise ValueError('Unexpected adapter name: '+target)
            if int(match.group(1)) == depth-1:
                if target.endswith('.B') and torch.count_nonzero(current[target]):
                    raise ValueError('Extra last-block adapter is not zero-delta')
                continue
            if source not in state:
                raise ValueError('Incomplete shared initializer: '+source)
            current[target] = state[source]
            used.add(source)
        if used != set(state):
            raise ValueError('Initializer contains unused or unrecognized parameters')
        load_adapter_state(self, current)
        self._shared_initializer_state = adapter_state(self)

    @contextmanager
    def initializer_reference(self):
        current = adapter_state(self)
        try:
            load_adapter_state(self, self._shared_initializer_state)
            yield
        finally:
            load_adapter_state(self, current)

    def assert_frozen(self):
        if frozen_digest(self) != self._frozen_digest:
            raise RuntimeError('A frozen base tensor changed')

    def save_trainable(self, path):
        torch.save(adapter_state(self), path)

    def load_trainable(self, path, reset_reference=False):
        if reset_reference:
            raise ValueError('Reference is always the pretrained actor with adapters disabled')
        load_adapter_state(self, torch.load(path, map_location='cpu', weights_only=True))

    def parameter_report(self):
        return dict(spec=asdict(self.spec), rank=self.rank, alpha=self.alpha,
            trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
            stored=sum(p.numel() for p in self.parameters()), target_modules=self.targets,
            frozen_sha256=self._frozen_digest, reference=('shared trained initializer' if
                self._shared_initializer_state is not None else 'pretrained actor; zero LoRA contribution'))
