class ConvLSTMCell(nn.Module):
    """
    adapted from: https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py
    belonging to the paper: " Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" by Shi et al.
    """

    def __init__(
        self, hidden_channels, kernel_size, input_channels=None, output_channels=None
    ):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0
        if input_channels is None:
            input_channels = hidden_channels
        if output_channels is None:
            output_channels = hidden_channels

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whi = nn.Conv2d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv2d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whf = nn.Conv2d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv2d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whc = nn.Conv2d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv2d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Who = nn.Conv2d(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        print(f"x size {x.size()}, h size {h.size()}, c size {c.size()}")
        print(self.Wxi(x).size())
        print(self.Whi(h).size())
        print((c * self.Wci).size())
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        print(f"ci {ci.size} cf {cf.size}")
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, shape, layer_type=None):
        if layer_type == "first":
            hidden = self.input_channels
        elif layer_type == "last":
            hidden = self.output_channels
        else:
            hidden = self.hidden_channels

        print("init hidden:", hidden)

        if self.Wci is None:
            self.Wci = nn.Parameter(
                torch.zeros(1, self.input_channels, shape[0], shape[1])
            )  # .cuda()
            self.Wcf = nn.Parameter(
                torch.zeros(1, hidden, shape[0], shape[1])
            )  # .cuda()
            self.Wco = nn.Parameter(
                torch.zeros(1, hidden, shape[0], shape[1])
            )  # .cuda()
        else:
            assert shape[0] == self.Wci.size()[2], "Input Height Mismatched!"
            assert shape[1] == self.Wci.size()[3], "Input Width Mismatched!"
        return (
            Variable(
                torch.zeros(batch_size, self.input_channels, shape[0], shape[1])
            ),  # .cuda()
            Variable(torch.zeros(batch_size, self.hidden_channels, shape[0], shape[1])),
        )  # .cuda())


class ConvLSTM(BaseModel):
    """
    adapted from: https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py
    belonging to the paper: " Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" by Shi et al.

    in the original implementation, c and h were allowed to have different channel dimensions in a layer, do not see how this works as they have to be added at some point...
    implementing with same channels per layer now
    (overall a bit goofy)

    #TODO: fix channel adaptation
    """

    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(
        self,
        input_channels,  # num features in
        hidden_channels,
        output_channels,  # num features out
        kernel_size,
        num_hidden_layers=2,
        step=1,  # predictive steps
        effective_step=[1],
    ):
        super(ConvLSTM, self).__init__()

        self.input_channels = input_channels  # [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels  # hidden_channels + [output_channels]
        self.output_channels = output_channels  # output_channels
        self.kernel_size = kernel_size
        self.num_layers = num_hidden_layers + 1
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []

        for i in range(self.num_layers):
            name = "cell{}".format(i)
            if i == 0:
                cell = ConvLSTMCell(
                    hidden_channels=self.hidden_channels,
                    input_channels=self.input_channels,
                    kernel_size=self.kernel_size,
                )
            elif i == (self.num_layers - 1):
                cell = ConvLSTMCell(
                    hidden_channels=self.hidden_channels,
                    output_channels=self.output_channels,
                    kernel_size=self.kernel_size,
                )
            else:
                cell = ConvLSTMCell(self.hidden_channels, self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, X: Tensor):  # X of size (batch_size, time, lon, lat, num_vars)
        # first -> pass input time step per time step through the convnet
        internal_state = []
        bsize, in_time, height, width, num_feats = X.size()

        for i, step in enumerate(range(in_time)):
            # get new input per time step
            x = X[:, i, :, :, :]
            # layers expect channels to come second
            x = torch.permute(x, (0, 3, 1, 2))
            print(f"Feeding step {step}, size {x.size()}")
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = "cell{}".format(i)
                if step == 0:
                    # bsize, _, height, width, num_feats = x.size()
                    if i == 0:
                        layer_type = "first"
                    elif i == (self.num_layers - 1):
                        layer_type = "last"
                    else:
                        layer_type = None
                    (h, c) = getattr(self, name).init_hidden(
                        batch_size=bsize, layer_type=layer_type, shape=(height, width)
                    )  # TODO: check output dimensionality!
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                print("h", h.size())
                print("c", c.size())
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            # if step in self.effective_step:
            #    outputs.append(x)

        # second -> use last state to predict desired time steps into the future

        internal_state_pred = []
        outputs = []
        for step in range(self.step):
            # x = X
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = "cell{}".format(i)

                if step == 0:
                    # use latest internal state
                    internal_state_pred.append(internal_state[-1])

                # do forward
                (h, c) = internal_state_pred[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state_pred[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs  # only outputting states -> list with len number effective steps
