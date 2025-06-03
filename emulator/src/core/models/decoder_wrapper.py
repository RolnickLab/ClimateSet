from emulator.src.core.models.basemodel import BaseModel

class DecoderWrapper(BaseModel):
    def __init__(self, model, multihead_decoder, channels_last=True, **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters()
        self.model = model
        self.multihead_decoder = multihead_decoder
        self.channels_last = channels_last

    def forward(self, x, model_num):
        # model num: (batch_size) (can be different for each batch item)
        out = self.model(x)

        if self.channels_last:
            out = out.permute((0, 1, 4, 2, 3))

        out = self.multihead_decoder(out, model_num)

        if self.channels_last:
            out = out.permute((0, 1, 3, 4, 2))
        out = out.nan_to_num()

        return out
