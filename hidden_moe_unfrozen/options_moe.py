"""MoE config for hidden_moe_unfrozen: backbone trainable by default (freeze_hidden_backbone=False)."""
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
        num_experts: int = 4,
        top_k: int = 1,
        balance_loss_weight: float = 0.04,
        balance_loss_start_weight: float = 0.005,
        balance_loss_warmup_epochs: int = 10,
        router_jitter_noise: float = 0.0,
        router_input_dropout: float = 0.0,
        router_z_loss_weight: float = 0.001,
        router_temperature_start: float = 1.4,
        router_temperature_end: float = 1.0,
        router_grad_clip_norm: float = 1.0,
        router_fp32: bool = True,
        load_penalty_weight: float = 0.0,
        load_penalty_type: str = "max",
        expert_use_group_norm: bool = False,
        expert_dropout: float = 0.0,
        expert_weight_decay: float = 1e-4,
        expert_init_offset_scale: float = 1e-3,
        init_hidden_checkpoint: str = "",
        freeze_hidden_backbone: bool = False,
        freeze_discriminator: bool = False,
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
        self.balance_loss_start_weight = balance_loss_start_weight
        self.balance_loss_warmup_epochs = balance_loss_warmup_epochs
        self.router_jitter_noise = router_jitter_noise
        self.router_input_dropout = router_input_dropout
        self.router_z_loss_weight = router_z_loss_weight
        self.router_temperature_start = router_temperature_start
        self.router_temperature_end = router_temperature_end
        self.router_grad_clip_norm = router_grad_clip_norm
        self.router_fp32 = router_fp32
        self.load_penalty_weight = load_penalty_weight
        self.load_penalty_type = load_penalty_type
        self.expert_use_group_norm = expert_use_group_norm
        self.expert_dropout = expert_dropout
        self.expert_weight_decay = expert_weight_decay
        self.expert_init_offset_scale = expert_init_offset_scale
        self.init_hidden_checkpoint = init_hidden_checkpoint
        self.freeze_hidden_backbone = freeze_hidden_backbone
        self.freeze_discriminator = freeze_discriminator

