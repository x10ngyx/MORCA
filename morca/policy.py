import torch
from .model import CNN, encode_normalized


class Policy:
    mode = "sea7"

    def __init__(self, checkpoint=None, *, sampling=False, seed=42, exploration=0.0, device="cuda"):
        self.device = torch.device(device)
        self.generator = torch.Generator().manual_seed(seed)
        self.sampling, self.exploration = (sampling, exploration)
        self.action_mode = "policy_categorical" if sampling else "policy_argmax"
        self.current_feature = None
        self.net = None
        if checkpoint:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.net = CNN("G1", 2).to(self.device).eval()
            self.net.load_state_dict(payload["policy_net"])
            self.normalizer = {k: v.to(self.device) for (k, v) in payload["normalizer"].items()}

    @torch.no_grad()
    def choose(self, state):
        if self.net is None:
            probability = 0.5
        else:
            raw = torch.cat((self.current_feature, state)) if state.numel() == 7 else state
            with torch.autocast(self.device.type, enabled=False):
                x = encode_normalized(raw.to(self.device), self.normalizer).float()
                logits = self.net(x.unsqueeze(0))[0]
                probability = float(logits.softmax(-1)[1])
            if not self.sampling:
                return (int(logits.argmax()), probability)
        probability = (1 - self.exploration) * probability + self.exploration * 0.5
        return (int(torch.rand((), generator=self.generator) < probability), probability)
