def blocks(self, x, kwargs):
    skip_forward = False
    ori_x = x
    if self.cnt >= int(self.num_steps * self.ret_ratio):
        test_x, test_kwargs = (x.clone(), kwargs)
        anchor_blocks = self.blocks[0 : self.probe_depth]
        for anchor_block in anchor_blocks:
            test_x = anchor_block(test_x, **test_kwargs)
        delta_x = (x - self.previous_input[self.cnt % 2]).abs().mean() / self.previous_input[
            self.cnt % 2
        ].abs().mean()
        delta_y = (
            test_x - self.previous_internal_states[self.cnt % 2]
        ).abs().mean() / self.previous_internal_states[self.cnt % 2].abs().mean()
        self.accumulated_rel_l1_distance[self.cnt % 2] += delta_y
        if self.accumulated_rel_l1_distance[self.cnt % 2] < self.rel_l1_thresh:
            skip_forward = True
            self.resume_flag[self.cnt % 2] = False
            residual_x = self.residual_cache[self.cnt % 2]
        else:
            self.resume_flag[self.cnt % 2] = True
            self.accumulated_rel_l1_distance[self.cnt % 2] = 0
    if skip_forward:
        ori_x = x.clone()
        if len(self.residual_window[self.cnt % 2]) >= 2:
            current_residual_indicator = test_x - x
            gamma = (
                (current_residual_indicator - self.probe_residual_window[self.cnt % 2][-2])
                .abs()
                .mean()
                / (
                    self.probe_residual_window[self.cnt % 2][-1]
                    - self.probe_residual_window[self.cnt % 2][-2]
                )
                .abs()
                .mean()
            ).clip(1, 2)
            x += self.residual_window[self.cnt % 2][-2] + gamma * (
                self.residual_window[self.cnt % 2][-1] - self.residual_window[self.cnt % 2][-2]
            )
        else:
            x = x + residual_x
        self.previous_internal_states[self.cnt % 2] = test_x
        self.previous_input[self.cnt % 2] = ori_x
    else:
        if self.resume_flag[self.cnt % 2]:
            x = test_x
            kwargs = test_kwargs
            unpass_blocks = self.blocks[self.probe_depth :]
        else:
            unpass_blocks = self.blocks
        for ind, block in enumerate(unpass_blocks):
            x = block(x, **kwargs)
            if ind == self.probe_depth - 1:
                if self.cnt >= int(self.num_steps * self.ret_ratio):
                    self.previous_internal_states[self.cnt % 2] = test_x
                else:
                    self.previous_internal_states[self.cnt % 2] = x
        residual_x = x - ori_x
        self.residual_cache[self.cnt % 2] = residual_x
        self.probe_residual_cache[self.cnt % 2] = (
            self.previous_internal_states[self.cnt % 2] - ori_x
        )
        self.previous_input[self.cnt % 2] = ori_x
        self.previous_output[self.cnt % 2] = x
        if len(self.residual_window[self.cnt % 2]) <= 2:
            self.residual_window[self.cnt % 2].append(residual_x)
            self.probe_residual_window[self.cnt % 2].append(self.probe_residual_cache[self.cnt % 2])
        else:
            self.residual_window[self.cnt % 2][-2] = self.residual_window[self.cnt % 2][-1]
            self.residual_window[self.cnt % 2][-1] = residual_x
            self.probe_residual_window[self.cnt % 2][-2] = self.probe_residual_window[self.cnt % 2][
                -1
            ]
            self.probe_residual_window[self.cnt % 2][-1] = self.probe_residual_cache[self.cnt % 2]
    return (x, skip_forward, locals().get("delta_y"), locals().get("gamma"))
