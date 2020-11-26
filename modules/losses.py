import torch

def mean_batch(val):
    mean = val.view(val.shape[0], -1).mean(-1)
    return  mean

def sum_batch(val):
    sum_b = val.view(val.shape[0], -1).sum(-1)
    return sum_b

def masked_l2_heatmap_loss(prediction, target, mask=None):
    #print(f"prediction {prediction.shape} target {target.shape}")
    diff = l2_loss(prediction,target)
    diff = diff.view(prediction.shape[0],prediction.shape[1], -1).mean(2)
    #print(f"diff.shape  {diff.shape}")

    if mask is not None:
        if (mask != mask).sum() > 0:
            print('NaN on mask!!')
            raise ValueError
        img_mask = torch.ones(diff.shape)
        # print(f"prediction.shape {prediction.shape}")
        # print(f"img_mask.shape {img_mask.shape}")
        # print(f"mask.shape {mask.shape}")
        # print(f"diff.shape  {diff.shape}")
        diff = diff * mask.squeeze(-1)
        #print(f"diff.shape  {diff.shape}")
        division = mask.sum(1) # .unsqueeze(-1)#.unsqueeze(-1)
        if (division == 0).sum() > 0:
            print('division by Zero!')
            raise ValueError
        #print(f"diff shape : {diff.shape}, division.shape {division.shape}")
        diff = (diff / division)
        diff = diff.view(diff.shape[0], -1).sum(-1)
        return diff
    return mean_batch(diff)


def masked_l2_loss(prediction, target, mask=None):
    diff = l2_loss(prediction,  target)
    if mask is not None:
        if (mask != mask).sum() > 0:
            print('NaN on mask!!')
            raise ValueError
        diff = diff * mask
        division = mask.sum(1).unsqueeze(-1)
        if (division == 0).sum() > 0:
            print('division by Zero!')
            raise ValueError
        diff = (diff / division)
        diff = diff.view(diff.shape[0], -1).sum(-1)
        return diff
    return mean_batch(diff)

def l2_loss(prediction, target, weight=1):
    if weight == 0:
        return 0
    return weight * torch.pow(prediction  - target,2)

def l1_loss(prediction, target, weight=1):
    if weight == 0:
        return 0
    return weight * sum_batch(torch.abs(prediction - target))
    #return weight * mean_batch(torch.abs(prediction - target))



def reconstruction_loss(prediction, target, weight):
    if weight == 0:
        return 0
    return weight * mean_batch(torch.abs(prediction - target))

def generator_gan_loss(discriminator_maps_generated, weight):
    scores_generated = discriminator_maps_generated
    score = (1 - scores_generated) ** 2
    return weight * mean_batch(score)

def discriminator_gan_loss(discriminator_maps_generated, discriminator_maps_real, weight):
    scores_real = discriminator_maps_real#[-1]
    scores_generated = discriminator_maps_generated#[-1]
    score = (1 - scores_real) ** 2 + scores_generated ** 2
    return weight * mean_batch(score)

def generator_loss_names(loss_weights):
    loss_names = []
    if loss_weights['reconstruction_deformed'] != 0:
        loss_names.append("rec_def")

    if loss_weights['reconstruction'] is not None:
        for i, _ in enumerate(loss_weights['reconstruction']):
            if loss_weights['reconstruction'][i] == 0:
                continue
            loss_names.append("layer-%s_rec" % i)

    loss_names.append("gen_gan")
    return loss_names


def discriminator_loss_names():
    return ['disc_gan']


def generator_loss(discriminator_maps_generated, discriminator_maps_real, video_deformed, loss_weights):
    loss_values = []
    if loss_weights['reconstruction_deformed'] != 0:
        loss_values.append(reconstruction_loss(discriminator_maps_real[0], video_deformed,
                                               loss_weights['reconstruction_deformed']))

    if loss_weights['reconstruction'] != 0:
        for i, (a, b) in enumerate(zip(discriminator_maps_real[:-1], discriminator_maps_generated[:-1])):
            if loss_weights['reconstruction'][i] == 0:
                continue
            loss_values.append(reconstruction_loss(b, a, 
                                                    weight=loss_weights['reconstruction'][i]))

    loss_values.append(generator_gan_loss(discriminator_maps_generated, 
                                           weight=loss_weights['generator_gan']))

    return loss_values

def generator_loss_2d(discriminator_maps_generated, discriminator_maps_real, loss_weights):
    loss_values = []
    if loss_weights['reconstruction'] != 0:
        for i, (a, b) in enumerate(zip(discriminator_maps_real[:-1], discriminator_maps_generated[:-1])):
            if loss_weights['reconstruction'][i] == 0:
                continue
            loss_values.append(reconstruction_loss(b, a, 
                                                    weight=loss_weights['reconstruction'][i]))

    loss_values.append(generator_gan_loss(discriminator_maps_generated, 
                                           weight=loss_weights['generator_gan']))

    return loss_values



def discriminator_loss(discriminator_maps_generated, discriminator_maps_real, loss_weights):
    loss_values = [discriminator_gan_loss(discriminator_maps_generated, 
                                           discriminator_maps_real,
                                           loss_weights['discriminator_gan'])]
    return loss_values
