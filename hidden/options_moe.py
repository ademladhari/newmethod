from options import HiDDenConfiguration, TrainingOptions


class HiDDenMoEConfiguration(HiDDenConfiguration):
    def __init__(
        self,
        H: int,
        W: int,
        message_length: int,
        encoder_blocks: int,
        encoder_channels: int,
        decoder_blocks: int,
        decoder_channels: int,
        use_discriminator: bool,
        use_vgg: bool,
        discriminator_blocks: int,
        discriminator_channels: int,
        decoder_loss: float,
        encoder_loss: float,
        adversarial_loss: float,
        enable_fp16: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        balance_loss_weight: float = 0.01,
    ):
        super(HiDDenMoEConfiguration, self).__init__(
            H=H,
            W=W,
            message_length=message_length,
            encoder_blocks=encoder_blocks,
            encoder_channels=encoder_channels,
            decoder_blocks=decoder_blocks,
            decoder_channels=decoder_channels,
            use_discriminator=use_discriminator,
            use_vgg=use_vgg,
            discriminator_blocks=discriminator_blocks,
            discriminator_channels=discriminator_channels,
            decoder_loss=decoder_loss,
            encoder_loss=encoder_loss,
            adversarial_loss=adversarial_loss,
            enable_fp16=enable_fp16,
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_weight = balance_loss_weight

