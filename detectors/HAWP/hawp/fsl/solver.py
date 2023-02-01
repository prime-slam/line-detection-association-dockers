import torch

def make_optimizer(cfg, model, loss_reducer = None):

    params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr=cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        # if 'md_predictor' in key or 'st_predictor' in key or 'ed_predictor' in key:
        #     lr = cfg.SOLVER.BASE_LR*100.0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if loss_reducer is not None:
        for key, value in loss_reducer.named_parameters():
            params += [{"params": [value], "lr": lr, "weight_decay": 0}]
        

    if cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(params,
                                    cfg.SOLVER.BASE_LR,
                                    momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER == 'ADAM':
        optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                     amsgrad=cfg.SOLVER.AMSGRAD)
    elif cfg.SOLVER.OPTIMIZER == 'ADAMW':
        optimizer = torch.optim.AdamW(params,
                                      cfg.SOLVER.BASE_LR,betas=(0.9,0.999),
                                      eps=1e-8,
                                      weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                      amsgrad=cfg.SOLVER.AMSGRAD,
        )
    else:
        raise NotImplementedError()
    return optimizer

def make_lr_scheduler(cfg,optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=cfg.SOLVER.STEPS,gamma=cfg.SOLVER.GAMMA)