"""Declarative, on-device parameter distributions for ECGImageLayer."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Uniform:
    low: object
    high: object


@dataclass(frozen=True)
class Choice:
    values: object


@dataclass(frozen=True)
class Bernoulli:
    probability: object


class ParameterSampler(nn.Module):
    """Compile constants to buffers once, then sample a whole batch in torch."""
    def __init__(self, distribution, prototype, resolve):
        super().__init__()
        self.tail = tuple(prototype.shape[1:])
        self.dtype = prototype.dtype
        if isinstance(distribution, Uniform):
            if not prototype.is_floating_point():
                raise ValueError('Uniform requires a continuous parameter. Use Choice for integer IDs')
            self.kind = 'uniform'
            low = torch.as_tensor(distribution.low, device=prototype.device, dtype=prototype.dtype).detach().expand_as(prototype).clone()
            high = torch.as_tensor(distribution.high, device=prototype.device, dtype=prototype.dtype).detach().expand_as(prototype).clone()
            if not torch.isfinite(low).all() or not torch.isfinite(high).all() or (high < low).any():
                raise ValueError('Uniform bounds must be finite with high >= low')
            self.register_buffer('low', low)
            self.register_buffer('high', high)
        elif isinstance(distribution, Bernoulli):
            self.kind = 'bernoulli'
            probability = torch.as_tensor(distribution.probability, device=prototype.device, dtype=torch.float32).detach().expand_as(prototype).clone()
            if not torch.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                raise ValueError('Bernoulli probability must be in [0,1]')
            self.register_buffer('probability', probability)
        elif isinstance(distribution, Choice):
            self.kind = 'choice'
            values = list(distribution.values)
            if not values:
                raise ValueError('Choice requires at least one value')
            bank = torch.stack([torch.as_tensor(resolve(value), device=prototype.device, dtype=prototype.dtype)
                                .detach().expand_as(prototype).squeeze(0) for value in values])
            if not torch.isfinite(bank).all():
                raise ValueError('Choice values must be finite')
            self.register_buffer('bank', bank)
        else:
            raise TypeError('Unsupported parameter distribution')

    def specification(self):
        return {'kind': self.kind, 'shape': self.tail, 'dtype': str(self.dtype),
                'buffer_shapes': {name: tuple(value.shape) for name, value in self.named_buffers()}}

    def forward(self, batch_size, generator=None):
        buffer = next(self.buffers())
        shape = (batch_size, *self.tail)
        if self.kind == 'choice':
            indices = torch.randint(len(self.bank), (batch_size,), device=buffer.device, generator=generator)
            return self.bank[indices]
        value = torch.rand(shape, device=buffer.device, generator=generator)
        if self.kind == 'bernoulli':
            return (value < self.probability).to(self.dtype)
        return (self.low + value * (self.high - self.low)).to(self.dtype)


def parameters_from_json(parameters):
    """Fixed JSON values or {uniform:[low,high]}, {choice:[...]}, {bernoulli:p}."""
    result = {}
    for key, value in parameters.items():
        if isinstance(value, dict):
            if set(value) == {'uniform'} and isinstance(value['uniform'], list) and len(value['uniform']) == 2:
                value = Uniform(*value['uniform'])
            elif set(value) == {'choice'} and isinstance(value['choice'], list):
                value = Choice(value['choice'])
            elif set(value) == {'bernoulli'}:
                value = Bernoulli(value['bernoulli'])
            else:
                raise ValueError(f'Invalid parameter distribution: {key}')
        result[key] = value
    return result
