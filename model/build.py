from .slowfast import SlowFast
from .vivit import ViViT


def build_model(config):
    model_arch = config.MODEL.ARCH

    if model_arch == "slowfast":
        model = SlowFast(config.MODEL.NUM_CLASSES)
    elif model_arch == "vivit":
        model = ViViT(num_classes=config.MODEL.NUM_CLASSES,
                      size=config.MODEL.VIVIT.INPUT_SIZE,
                      frame_per_clip=config.MODEL.VIVIT.FRAME_PER_CLIP,
                      t=config.MODEL.VIVIT.T,
                      h=config.MODEL.VIVIT.H,
                      w=config.MODEL.VIVIT.W,
                      n_head=config.MODEL.VIVIT.NUM_HEAD,
                      n_layer=config.MODEL.VIVIT.NUM_LAYER,
                      d_model=config.MODEL.VIVIT.D_MODEL,
                      d_feature=config.MODEL.VIVIT.D_FEATURE,
                      use_checkpoint=config.MODEL.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"{model_arch}")

    return model
