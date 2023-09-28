#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random


class Vocab(object):
    def __init__(self, empty_init=False):
        if empty_init:
            self.stoi, self.itos, self.vocab_sz = {}, [], 0
        else:
            self.stoi = {
                w: i
                for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
            }
            self.itos = [w for w in self.stoi]
            self.vocab_sz = len(self.itos)
        self.words = []
        for w in self.stoi:
            self.words.append(w)

    def get_rand_word(self):
        return random.choice(self.words)

    def add(self, words):
        cnt = len(self.itos)
        for word in words:
            if word in self.stoi:
                continue
            self.stoi[word] = cnt
            self.itos.append(word)
            cnt += 1
        self.vocab_sz = len(self.itos)