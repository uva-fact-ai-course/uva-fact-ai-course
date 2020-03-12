"""
Hard-debias and soft-debias for word embeddings.
Extended from the code from:

Man is to Computer Programmer as Woman is to Homemaker?
    Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

import numpy as np
import torch

from . import we


def hard_debias(E, gender_specific_words, definitional, equalize):
    """
    Hard debiases word embeddings.

    :param object E: WordEmbedding object.
    :param list gender_specific_words: List of gender specific words, which are
        not dibiased.
    :param list definitional: List containing lists of corresponding
        definitional words.
    :param list equalize: List containing lists of corresponding words that
        should only differ in gender.
    """
    gender_direction = we.doPCA(definitional, E).components_[0]
    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = we.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()


def soft_debias(
        E, gender_specific_words, defs, lamb=0.2,
        log=True, print_every=100, epochs=2000, lr=0.01, gamma=0.1,
        decrease_times=None):
    """
    Soft debiases word embeddings.


    :param object E: WordEmbedding object.
    :param list gender_specific_words: List of gender specific words, which are
        not debiased.
    :param list defs: List containing lists of corresponding
        definitional words.
    :param float lamb: Lambda value for soft debiasing.
    :param bool log: Print optimizer progress.
    :param int print_every: If `log` is True, print loss every `print_every`
        epochs.
    :param int epochs: Number of epochs to fit transformation matrix T.
    :param float lr: Learning rate for Adam optimizer.
    :param float gamma: Multiplicative factor of learning rate decay.
    :param list decrease_times: Epoch numbers to decrease learning rate.
    """
    if decrease_times is None:
        decrease_times = [1000, 1500, 1800]
    W = torch.from_numpy(E.vecs).t()
    dim = W.shape[0]
    neutrals = list(set(E.words) - set(gender_specific_words))
    neutrals = torch.tensor([E.vecs[E.index[w]] for w in neutrals]).t()
    gender_direction = torch.tensor([we.doPCA(defs, E).components_[0]]).t()

    L = lamb  # lambda
    u, s, _ = torch.svd(W)
    s = torch.diag(s)

    # precompute
    t1 = s.mm(u.t())
    t2 = u.mm(s)

    transform = torch.randn(dim, dim, requires_grad=True)
    epochs = epochs
    optimizer = torch.optim.Adam([transform], lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decrease_times, gamma=gamma)
    best = (None, float("inf"))  # (best transform, lowest loss)

    for i in range(epochs):
        optimizer.zero_grad()
        TtT = torch.mm(transform.t(), transform)
        norm1 = (t1.mm(TtT - torch.eye(dim)).mm(t2)).norm(p="fro")
        norm2 = (neutrals.t().mm(TtT).mm(gender_direction)).norm(p="fro")
        loss = norm1 + L * norm2
        if loss.item() < best[1]:
            best = (transform, loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        if log and i % print_every == 0:
            print("Loss @ Epoch #" + str(i) + ":", loss.item())

    transform = best[0].detach()
    if log:
        print(f"Lowest loss: {best[1]}")

    debiased_embeds = transform.mm(W).t().numpy()
    debiased_embeds = debiased_embeds / np.linalg.norm(
        debiased_embeds, axis=1)[:, None]
    E.vecs = debiased_embeds
