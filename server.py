# see https://github.com/Lightning-AI/litserve?tab=readme-ov-file#serve-on-gpus
from fastapi import Request, Response

import torch
import torch.nn as nn

from litserve import LitAPI, LitServer

from config import Config, ModuleConfig
from module import SequenceClassificationModule


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        # load and move the model to the correct device
        self.lit_module = SequenceClassificationModule.load_from_checkpoint(ModuleConfig.finetuned).to(device)
        # keep track of the device for moving data accordingly
        self.device = device

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, sequence):
        return self.lit_module.predict_step(sequence)

    def encode_response(self, output) -> Response:
        return {"output": output}


if __name__ == "__main__":
    server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=1, timeout=30)
    server.run(port=8000)
