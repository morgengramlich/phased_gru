'''Implementaton of phased GRU layer'''

import torch
from torch import nn
import torch.jit as jit

class PhasedGRUCell(jit.ScriptModule):
    '''Phased gated recurrent unit cell'''

    def __init__(self, input_size, hidden_size, leak_rate = 0.001):
        super(PhasedGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )
        self.period = nn.parameter.Parameter(torch.randn(hidden_size, 1))
        self.open_ratio = nn.parameter.Parameter(torch.randn(hidden_size, 1))
        self.phase_shift = nn.parameter.Parameter(torch.randn(hidden_size, 1))
        self.leak_rate = leak_rate

    @jit.script_method
    def forward(self, x, timestamp, hx = None):
        '''One step of computation'''
        if hx is None:
            hx = self._init_state()
        hx_hat = self.gru_cell(x, hx)
        kt = self._time_gate(timestamp)
        hx_t = kt * hx_hat + (1.0 - kt) * hx
        return hx_t

    def _leak_rate(self):
        return self.leak_rate if self.training else 0.0

    def _cycle_phase(self, timestamp):
        return torch.div(torch.fmod(timestamp - self.phase_shift, self.period), self.period)

    def _time_gate(self, timestamp):
        cycle_phase = self._cycle_phase(timestamp)
        up_phase_value = 2 * cycle_phase / self.open_ratio
        down_phase_value = 2 - up_phase_value
        closed_phase_value = self._leak_rate() * cycle_phase

        result = torch.where(
            cycle_phase < 0.5 * self.open_ratio,
            up_phase_value,
            torch.where(
                cycle_phase < self.open_ratio,
                down_phase_value,
                closed_phase_value
            )
        )

        return torch.transpose(result, 0, 1)

    def _init_state(self):
        return torch.zeros(1, self.hidden_size)


class PhasedGRU(jit.ScriptModule):
    '''Multi-layer phased gated recurrent unit'''

    def __init__(self, input_size, hidden_size, num_layers = 1, leak_rate = 0.001):
        super(PhasedGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([PhasedGRUCell(input_size, hidden_size, leak_rate)])
        for _ in range(num_layers - 1):
            self.cells.append(PhasedGRUCell(hidden_size, hidden_size, leak_rate))

    @jit.script_method
    def forward(self, values, timestamps, state = None):
        '''One step of computation'''
        if state is None:
            state = self._init_state(values.shape[0])

        top_states = []
        final_states = []
        for seq_idx in range(values.shape[1]):
            step_value = values.select(1, seq_idx)
            step_timestamps = timestamps.select(1, seq_idx)
            for layer_idx, cell in enumerate(self.cells):
                h = cell(step_value, step_timestamps, state[layer_idx])
                step_value = h
                if layer_idx == self.num_layers - 1:
                    top_states.append(h)
                if seq_idx == values.shape[1] - 1:
                    final_states.append(h)
        output = torch.stack(top_states, dim=1)
        final_state = torch.stack(final_states, dim=0)
        return output, final_state

    def _init_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
