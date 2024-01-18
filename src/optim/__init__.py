import torch.nn as nn


def define_criterion(criterion_opt):
    if criterion_opt is None:
        return None, None

    # parse
    if criterion_opt["type"] == "MSE":
        criterion = nn.MSELoss(reduction=criterion_opt["reduction"])

    elif criterion_opt["type"] == "L1":
        criterion = nn.L1Loss(reduction=criterion_opt["reduction"])

    elif criterion_opt["type"] == "CB":
        from .losses import CharbonnierLoss

        criterion = CharbonnierLoss(reduction=criterion_opt["reduction"])

    elif criterion_opt["type"] == "CosineSimilarity":
        from .losses import CosineSimilarityLoss

        criterion = CosineSimilarityLoss()

    elif criterion_opt["type"] == "GAN":
        from .losses import VanillaGANLoss

        criterion = VanillaGANLoss(reduction=criterion_opt["reduction"])

    elif criterion_opt["type"] == "LSGAN":
        from .losses import LSGANLoss

        criterion = LSGANLoss(reduction=criterion_opt["reduction"])

    elif criterion_opt["type"] == "LPIPS":
        import lpips

        criterion = lpips.LPIPS(net=criterion_opt["net"])

    else:
        raise ValueError("Unrecognized criterion: {}".format(criterion_opt["type"]))

    return criterion, criterion_opt["weight"]
