# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 23:18:51 2019

@author: saughosh
"""

from word2veclite import Word2Vec
import matplotlib.pyplot as plt
import numpy as np

corpus = "I like playing football with my friends"
y = ["I like playing criket with my sister"]
skipgram = Word2Vec(method="skipgram", corpus=corpus,
                window_size=1, n_hidden=2,
                n_epochs=600, learning_rate=0.1)
W1, W2, loss_vs_epoch = skipgram.run()

#plt.yticks(range(0,24,2))
#plt.ylim([6,24])
plt.plot(loss_vs_epoch)
plt.show()