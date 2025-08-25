"""
    This script defines different types of custom loss functions that might be used
    for training the CVRL model.

    Author: Ali Ghelmani,       Date: Dec. 15, 2021
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import toeplitz


def contrastive_loss(batch, config):
    """
        This function implements the contrastive loss used in the CVRL paper. Batch correspond to the
        output of the network for different augmented versions of clips that pass through the model.
        The positive pairs are different augmented version of the same clip while the remaining
        clips in the batch are treated as negative samples.
    :param batch: The concatenated output of the two streams in the model.
    :param config: Only need the temperature parameter for now!
    :return: The computed loss for the batch.
    """
    cosine_sim = nn.CosineSimilarity(dim=2)
    loss = nn.CrossEntropyLoss(reduction="sum")
    batch_size = config.train_cfg.train_batch_size

    mask = mask_correlated_samples(batch_size)
    mask = mask.to(batch.device)

    """
        A bit hard to wraps one's head around! For a batch of model output
        with dimensions of (N, D) it creates a matrix of cosine similarities between different data
        samples with the end result being a symmetric matrix of the shape (N x N).
    """
    similarity = cosine_sim(batch.unsqueeze(1), batch.unsqueeze(0)) / config.train_cfg.temperature

    sim_i_j = torch.diag(similarity, batch_size)
    sim_j_i = torch.diag(similarity, -batch_size)

    # the above diags extract the similarity output for augmented pairs
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)
    negative_samples = similarity[mask].reshape(2 * batch_size, -1)

    # A zeros vector of size 2N, because with the following cat operation the positives end up at the zero location
    labels = torch.zeros(2 * batch_size).to(positive_samples.device).long()
    logits = torch.cat([positive_samples, negative_samples], dim=1)

    return loss(logits, labels) / (2 * batch_size)


def mask_correlated_samples(batch_size):
    """
        This function creates a mask of the form:
            [[0, 1, 1, 0, 1],
             [1, 0, 1, 1, 0],
             [0, 1, 0, 1, 1],
             [1, 0, 1, 0, 1],
             [1, 1, 0, 1, 0]]
        which is of Toeplitz form.
    :param batch_size: ...
    :return: A masking matrix of the above form.
    """
    n = 2 * batch_size
    mask = np.ones((1, n), dtype=bool)
    mask[0, 0] = 0
    mask[0, batch_size] = 0
    mask = torch.from_numpy(toeplitz(mask))
    return mask
