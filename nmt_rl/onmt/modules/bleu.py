#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentence level and Corpus level BLEU score calculation tool
From https://github.com/cshanbo/Smooth_BLEU/blob/master/bleu.py
"""

from __future__ import division, print_function

import io
import os
import math
import sys
import argparse
from fractions import Fraction
from collections import Counter
from functools import reduce
from operator import or_
import torch
import numpy as np

try:
    from nltk import ngrams
except:
    def ngrams(sequence, n):
        sequence = iter(sequence)
        history = []
        while n > 1:
            history.append(next(sequence))
            n -= 1
        for item in sequence:
            history.append(item)
            yield tuple(history)
            del history[0]


def modified_precision(references, hypothesis, n):
    # Extracts all ngrams in hypothesis.
    counts = Counter(ngrams(hypothesis, n))
    if not counts:
        return Fraction(0)
    # Extract a union of references' counts.
    max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()}
    return Fraction(sum(clipped_counts.values()), sum(counts.values()))


def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                segment_level=False, smoothing=0, epsilon=1, alpha=1,
                k=5):
    # Initialize the numbers.
    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0
    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lens = (len(reference) for reference in references)
        closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
        ref_lengths += closest_ref_len
        # Calculates the modified precision for each order of ngram.
        segment_level_precision = []
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator
            segment_level_precision.append(p_i)

        # Optionally, outputs segment level scores.
        if segment_level:
            if hyp_len == 0:
                print(0)
            else:
                _bp = min(math.exp(1 - closest_ref_len / hyp_len), 1.0)
                segment_level_precision = chen_and_cherry(references, hypothesis,
                                                          segment_level_precision,
                                                          hyp_len, smoothing, epsilon,
                                                          alpha)
                segment_pn = [w * math.log(p_i) if p_i != 0 else 0 for p_i, w in
                              zip(segment_level_precision, weights)]
                print(_bp * math.exp(math.fsum(segment_pn)))

    # Calculate corpus-level brevity penalty.
    bp = min(math.exp(1 - ref_lengths / hyp_lengths), 1.0)

    # Calculate corpus-level modified precision.
    p_n = []
    p_n_str = []
    for i, w in enumerate(weights, start=1):
        p_i = Fraction(p_numerators[i] / p_denominators[i])
        p_n_str.append(p_i)
        try:
            p_n.append(w * math.log(p_i))
        except ValueError:
            p_n.append(0)

    # Final bleu score.
    score = bp * math.exp(math.fsum(p_n))
    return score, p_n_str, hyp_lengths, ref_lengths


def chen_and_cherry(references, hypothesis, p_n, hyp_len,
                    smoothing=0, epsilon=0.1, alpha=5, k=5):
    """
    Boxing Chen and Collin Cherry (2014) A Systematic Comparison of Smoothing
    Techniques for Sentence-Level BLEU. In WMT14.
    """
    # No smoothing.
    if smoothing == 0:
        return p_n
    # Smoothing method 1: Add *epsilon* counts to precision with 0 counts.
    if smoothing == 1:
        return [Fraction(p_i.numerator + epsilon, p_i.denominator)
                if p_i.numerator == 0 else p_i for p_i in p_n]
    # Smoothing method 2: Add 1 to both numerator and denominator (Lin and Och 2004)
    if smoothing == 2:
        return [Fraction(p_i.numerator + 1, p_i.denominator + 1)
                for p_i in p_n]
    # Smoothing method 3: NIST geometric sequence smoothing
    # The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each
    # precision score whose matching n-gram count is null.
    # k is 1 for the first 'n' value for which the n-gram match count is null/
    # For example, if the text contains:
    #   - one 2-gram match
    #   - and (consequently) two 1-gram matches
    # the n-gram count for each individual precision score would be:
    #   - n=1  =>  prec_count = 2     (two unigrams)
    #   - n=2  =>  prec_count = 1     (one bigram)
    #   - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
    #   - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
    if smoothing == 3:
        incvnt = 1  # From the mteval-v13a.pl, it's referred to as k.
        for i, p_i in enumerate(p_n):
            if p_i == 0:
                p_n[i] = 1 / 2 ** incvnt
                incvnt += 1
        return p_n
    # Smoothing method 4:
    # Shorter translations may have inflated precision values due to having
    # smaller denominators; therefore, we give them proportionally
    # smaller smoothed counts. Instead of scaling to 1/(2^k), Chen and Cherry
    # suggests dividing by 1/ln(len(T), where T is the length of the translation.
    if smoothing == 4:
        incvnt = 1
        for i, p_i in enumerate(p_n):
            if p_i == 0:
                p_n[i] = incvnt * k / math.log(hyp_len)  # Note that this K is different from the K from NIST.
                incvnt += 1
        return p_n
    # Smoothing method 5:
    # The matched counts for similar values of n should be similar. To a
    # calculate the n-gram matched count, it averages the n−1, n and n+1 gram
    # matched counts.
    if smoothing == 5:
        m = {}
        # Requires an precision value for an addition ngram order.
        p_n_plus5 = p_n + [modified_precision(references, hypothesis, 5)]
        m[-1] = p_n[0] + 1
        for i, p_i in enumerate(p_n):
            p_n[i] = (m[i - 1] + p_i + p_n_plus5[i + 1]) / 3
            m[i] = p_n[i]
        return p_n
    # Smoothing method 6:
    # Interpolates the maximum likelihood estimate of the precision *p_n* with
    # a prior estimate *pi0*. The prior is estimated by assuming that the ratio
    # between pn and pn−1 will be the same as that between pn−1 and pn−2.
    if smoothing == 6:
        for i, p_i in enumerate(p_n):
            if i in [1, 2]:  # Skips the first 2 orders of ngrams.
                continue
            else:
                pi0 = p_n[i - 1] ** 2 / p_n[i - 2]
                # No. of ngrams in translation.
                l = sum(1 for _ in ngrams(hypothesis, i + 1))
                p_n[i] = (p_i + alpha * pi0) / (l + alpha)
        return p_n
    # Smoothing method
    if smoothing == 7:
        p_n = chen_and_cherry(references, hypothesis, p_n, hyp_len, smoothing=4)
        p_n = chen_and_cherry(references, hypothesis, p_n, hyp_len, smoothing=5)
        return p_n


def sentence_bleu_nbest(reference, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing=0, epsilon=0.1, alpha=5, k=5):
    """
    TODO: support dynamic length
    :param reference: a list of words, like ['hallo' , ',', 'world']
    :param hypotheses: a list of words, like ['hallo' , ',', 'world']
    :return: a float
    """
    bleu_output = corpus_bleu([(reference,)], [hypotheses], weights)
    bleu_score, p_n, hyp_len, ref_len = bleu_output
    p_n = chen_and_cherry(reference, hypotheses, p_n, hyp_len, smoothing, epsilon)
    segment_pn = [w * math.log(p_i) if p_i != 0 else 0 for p_i, w in
                  zip(p_n, weights)]
    _bp = min(math.exp(1 - ref_len / hyp_len), 1.0)
    return _bp * math.exp(math.fsum(segment_pn))


def batch_bleu(reference, hypotheses):
    """
    Compute reward of every step in a batch.
    r_t = bleu_t - bleu_{t-1}, bleu_{-1} = 0

    :param reference: A index tensor of size (seq, b)
    :param hypotheses: A index tensor of size (seq, b)
    :return: rewards_tensor_batch: A tensor of size (seq, b)
    """
    _, batch_size = reference.size()

    # Convert to a list of a batch lists of int
    reference = reference.view(batch_size, -1).data.numpy().tolist()
    hypotheses = hypotheses.view(batch_size, -1).data.numpy().tolist()

    rewards_batch = []
    for ref_list, hyp_list in zip(reference, hypotheses):
        # A list of int, convert to a list of str for bleu function
        ref_list = list(map(str, ref_list))
        hyp_list = list(map(str, hyp_list))
        bleus_step = []
        len_hyp = len(hyp_list)
        for i in range(1, len_hyp+1):
            bleu_score = sentence_bleu_nbest(ref_list, hyp_list[:i])
            bleus_step.append(bleu_score)

        rewards_step = []
        for i in range(len(bleus_step)):
            if i == 0:
                rewards_step.append(bleus_step[i])
            else:
                rewards_step.append(bleus_step[i] - bleus_step[i-1])

        rewards_tensor = torch.from_numpy(np.array(rewards_step))
        rewards_tensor = torch.autograd.Variable(rewards_tensor) # (seq_len)
        rewards_batch.append(rewards_tensor)

    rewards_tensor_batch = torch.stack(rewards_batch, dim=1) # (seq_len, batch)
    return rewards_tensor_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for calculating BLEU')
    parser.add_argument('-t', '--translation', type=str, required=True,
                        help="translation file or string")
    parser.add_argument('-r', '--reference', type=str, required=True,
                        help="reference file or string")
    parser.add_argument('-s', '--smooth', type=int, default=3, metavar='INT', required=False,
                        help="smoothing method type (default: %(default)s)")
    parser.add_argument('-w', '--weights', type=str, default='0.25 0.25 0.25 0.25',
                        help="weights for ngram (default: %(default)s)")
    parser.add_argument('-sl', '--sentence-level', action='store_true',
                        help="print sentence level BLEU score (default: %(default)s)")
    parser.add_argument('-se', '--smooth-epsilon', type=float, default=0.1,
                        help="empirical smoothing parameter for method 1 (default: %(default)s)")
    parser.add_argument('-sk', '--smooth-k', type=int, default=5,
                        help="empirical smoothing parameter for method 4 (default: %(default)s)")
    parser.add_argument('-sa', '--smooth-alpha', type=int, default=5,
                        help="empirical smoothing parameter for method 6 (default: %(default)s)")

    args = parser.parse_args()

    hypothesis_file = args.translation
    reference_file = args.reference
    weights = tuple(map(float, args.weights.split()))
    segment_level = args.sentence_level
    smoothing_method = args.smooth
    epsilon = args.smooth_epsilon
    alpha = args.smooth_alpha
    k = args.smooth_k

    # Calculate BLEU scores.
    # Set --sentence-level and other params to calc sentence-level BLEU in a FILE or string
    if os.path.isfile(reference_file):
        with io.open(reference_file, 'r', encoding='utf8') as reffin, \
                io.open(hypothesis_file, 'r', encoding='utf8') as hypfin:
            list_of_references = ((r.split(),) for r in reffin)
            hypotheses = (h.split() for h in hypfin)
            corpus_bleu(list_of_references, hypotheses,
                        weights=weights, segment_level=segment_level,
                        smoothing=smoothing_method, epsilon=epsilon, alpha=alpha, k=k)
    else:
        reffin = [reference_file]
        hypfin = [hypothesis_file]
        list_of_references = ((r.split(),) for r in reffin)
        hypotheses = (h.split() for h in hypfin)
        corpus_bleu(list_of_references, hypotheses,
                    weights=weights, segment_level=True,
                    smoothing=smoothing_method, epsilon=epsilon, alpha=alpha, k=k)
