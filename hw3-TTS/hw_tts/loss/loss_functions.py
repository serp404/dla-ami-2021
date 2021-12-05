import torch


def fastspeech_loss(l2_weight=1., l1_weight=1.):
    l2_criterion = torch.nn.MSELoss()
    l1_criterion = torch.nn.L1Loss()

    def fs_criterion(mels_pred, mels_true, durs_pred, durs_true):
        loss = l2_weight * l2_criterion(mels_pred, mels_true) + \
            l1_weight * l1_criterion(durs_pred, durs_true)
        return loss

    return fs_criterion
