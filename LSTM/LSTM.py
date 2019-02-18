import numpy as np

"""
An overly verbose and basic LSTM script built for learning purposes
"""


class SoftMaxCrossEntropy(object):
    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def tangent_h(z):
        return np.tanh(z)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


class LSTM(object):
    def __init__(self, hidden, len_dic, conv2ix, conv2ch, functions=SoftMaxCrossEntropy):
        """

        :param hidden: size of the hidden layer
        :param len_dic: length of dictionary (# of unique chars)
        :param conv2ix: a dictionary that maps a char to a number
        :param conv2ch: a dictionary that maps a number to a char
        :param functions: class that contains functions which will be used for propagation
        """
        self.dim = len_dic
        self.hid = hidden
        self.fxn = functions
        self.c2ix = conv2ix
        self.c2ch = conv2ch

        # cache is used to hold all the calculated values while forward propagating for use in back-propagation
        self.cache = []

        # initializes weights related to input
        self.wfx, self.wix, self.wcx, self.wox = [np.random.randn(self.hid, len_dic) /
                                                  np.sqrt(len_dic) for x in range(4)]

        # initializes weights related to previous output
        self.wfh, self.wih, self.wch, self.woh = [np.random.randn(self.hid, self.hid) /
                                                  np.sqrt(len_dic) for x in range(4)]

        # initializes weights related to current output
        self.wy = np.random.randn(len_dic, self.hid) / np.sqrt(len_dic)

        # initializes biases
        self.bf, self.bi, self.bc, self.bo = [np.zeros((self.hid, 1)) for x in range(4)]
        self.by = np.zeros((len_dic, 1))

    def update_gradient(self, data, seq_length, eta=1):
        """
        Initiates update of gradients with samples provided when either a new best lost is achieved or if
        rep count reaches a multiple of 500

        :param data: the full input string
        :param seq_length: the number of cells to 'unroll' over
        :param eta: learning rate
         """
        p, rep = 0, 0
        stagnant_count = 0
        temp_loss = -np.log(1 / self.dim) * seq_length  # assume loss based on expected value from guess

        while True:
            # h_prev and c_prev if at the start of a new run or re-setting a run at the end of a sweep
            if p + seq_length + 1 >= len(data) or p == 0:
                h_prev = np.zeros((self.hid, 1))
                c_prev = np.zeros((self.hid, 1))
                state = [(h_prev, c_prev)]
                p = 0

            # get the input sequence and the corresponding output for each char
            ipt = data[p:p + seq_length]
            opt = data[p + 1:p + seq_length + 1]

            # convert char into numeric repesentation
            ipt = [self.c2ix[ch] for ch in ipt]
            opt = [self.c2ix[ch] for ch in opt]

            for n in range(seq_length):
                # vectorize input
                x_t = np.zeros((self.dim, 1))
                x_t[ipt[n]] = 1

                _, n_state = self.feedforward(x_t, state[-1])
                state.append(n_state)

            sum_dwy, sum_dby, sum_dwox, sum_dwoh, sum_dbo, sum_dwix, sum_dwih, sum_dbi, sum_dwcx, sum_dwch, \
                sum_dbc, sum_dwfx, sum_dwfh, sum_dbf, total_loss = self.backprop(opt, state)

            # simple learning schedule + results print out
            if total_loss > temp_loss:
                stagnant_count += 1

                if stagnant_count >= 5000:
                    eta /= 5
                    stagnant_count = 0
                    print("eta now: {0}".format(eta))
            else:
                temp_loss = total_loss

                sample_ix = self.sample(state, ipt[0], 200)
                txt = ''.join(self.c2ch[ix] for ix in sample_ix)
                print("Best loss sample: \n----\n {} \n----".format(txt))

                print("Best loss: {0}".format(total_loss))

                stagnant_count = 0

            self.wfx = self.wfx - eta * sum_dwfx / seq_length
            self.wix = self.wix - eta * sum_dwix / seq_length
            self.wcx = self.wcx - eta * sum_dwcx / seq_length
            self.wox = self.wox - eta * sum_dwox / seq_length
            self.wfh = self.wfh - eta * sum_dwfh / seq_length
            self.wih = self.wih - eta * sum_dwih / seq_length
            self.wch = self.wch - eta * sum_dwch / seq_length
            self.woh = self.woh - eta * sum_dwoh / seq_length
            self.wy = self.wy - eta * sum_dwy / seq_length
            self.bf = self.bf - eta * sum_dbf / seq_length
            self.bi = self.bi - eta * sum_dbi / seq_length
            self.bc = self.bc - eta * sum_dbc / seq_length
            self.bo = self.bo - eta * sum_dbo / seq_length
            self.by = self.by - eta * sum_dby / seq_length

            # print results every 500 reps
            if rep % 500 == 0:
                sample_ix = self.sample(state, ipt[0], 200)
                txt = ''.join(self.c2ch[ix] for ix in sample_ix)
                print("----\n {} \n----".format(txt))

                print(f"Total training lost at rep {rep} was: {total_loss}")

            self.cache = []
            rep += 1
            p += seq_length

    def backprop(self, opt, states):
        """
        Back-propagation of results

        :param opt: output values, given as an array of numbers which encode for the correct char
        :param states: an array of tuples that represent h_prev and c_prev for all cell states over propgation period
        :return: a lot of aggregated gradients and total loss
        """
        # initiates 0-matrix for initial dh_next and dc_next
        dh_next = np.zeros((self.hid, 1))
        dc_next = np.zeros((self.hid, 1))

        # initial aggregators for different weights
        sum_dwy, sum_dby, sum_dwox, sum_dwoh, sum_dbo, sum_dwix, sum_dwih, sum_dbi, sum_dwcx, sum_dwch, sum_dbc, \
            sum_dwfx, sum_dwfh, sum_dbf = [0 for x in range(14)]

        # loss accumulator
        total_loss = 0

        # the first state is a zero matrix for both h_prev and c_prev and is ignored
        for n in range(1, len(opt) + 1):
            h_prev, c_prev = states[-n - 1]

            ipt, zf, ft, zi, hi, zc, hc, zo, ho, ct, ht, zl, y = self.cache[-n]

            # y is an array of probabilities, opt[-n] gives the numerical encoding of the correct output
            total_loss += -np.log(y[opt[-n]])  # cross entopy loss === log of expected value for correct answer

            dy = y.copy()
            dy[opt[-n]] -= 1

            # calculate dC @ output layer
            dby = dy
            dwy = np.dot(dy, np.transpose(ht))

            # change in cost w.r.t. to historic output. Because cell state is passed on, it will
            # have an additional gradient component
            dh = np.dot(np.transpose(self.wy), dy) + dh_next

            # change in cost w.r.t. to current cell state. Because cell state is passed on, it will
            # have an additional gradient component
            dc = dh * ho * (1 - (np.tanh(ct) * np.tanh(ct))) + dc_next

            # calculate change in cost due to work/bias for output
            dzo = dh * np.tanh(ct) * (self.fxn.sigmoid(zo) * (1 - self.fxn.sigmoid(zo)))
            dwox = np.dot(dzo, np.transpose(ipt))
            dwoh = np.dot(dzo, np.transpose(h_prev))
            dbo = dzo

            # calculate change in cost due to work/bias for input gate
            dzi = dc * hc * (self.fxn.sigmoid(zi) * (1 - self.fxn.sigmoid(zi)))
            dwix = np.dot(dzi, np.transpose(ipt))
            dwih = np.dot(dzi, np.transpose(h_prev))
            dbi = dzi

            dzc = dc * hi * (1 - (np.tanh(zc) * np.tanh(zc)))
            dwcx = np.dot(dzc, np.transpose(ipt))
            dwch = np.dot(dzc, np.transpose(h_prev))
            dbc = dzc

            # calculate change in cost due to work/bias for forget gate
            dzf = dc * c_prev * (self.fxn.sigmoid(zf) * (1 - self.fxn.sigmoid(zf)))
            dwfx = np.dot(dzf, np.transpose(ipt))
            dwfh = np.dot(dzf, np.transpose(h_prev))
            dbf = dzf

            # dh_next needs to be aggregated across all gates
            dh_next = \
                np.dot(np.transpose(self.wfh), dzf) + np.dot(np.transpose(self.wih), dzi) + \
                np.dot(np.transpose(self.wch), dzc) + np.dot(np.transpose(self.woh), dzo)
            dc_next = dc * ft

            # aggregate gradients across cells
            sum_dwy += dwy
            sum_dby += dby

            sum_dwox += dwox
            sum_dwoh += dwoh
            sum_dbo += dbo

            sum_dwix += dwix
            sum_dwih += dwih
            sum_dbi += dbi

            sum_dwcx += dwcx
            sum_dwch += dwch
            sum_dbc += dbc

            sum_dwfx += dwfx
            sum_dwfh += dwfh
            sum_dbf += dbf

        # clip gradients to prevent explosion getting out of hand
        for dparam in [sum_dwy, sum_dby, sum_dwox, sum_dwoh, sum_dbo, sum_dwix, sum_dwih, sum_dbi, sum_dwcx, sum_dwch,
                       sum_dbc, sum_dwfx, sum_dwfh, sum_dbf]:
            np.clip(dparam, -5, 5, out=dparam)

        return \
            sum_dwy, sum_dby, sum_dwox, sum_dwoh, sum_dbo, sum_dwix, sum_dwih, sum_dbi, sum_dwcx, sum_dwch, \
            sum_dbc, sum_dwfx, sum_dwfh, sum_dbf, total_loss

    def feedforward(self, ipt, state):
        """
        Feed forward for LSTM
        :param ipt: vectorized input
        :param state: value of h_prev and c_prev
        :return: next char output and new state
        """
        h_prev, c_prev = state

        # calculates forget gate layer activation
        zf = np.dot(self.wfx, ipt) + np.dot(self.wfh, h_prev) + self.bf
        ft = self.fxn.sigmoid(zf)

        # calculates input gate layer activation
        zi = np.dot(self.wix, ipt) + np.dot(self.wih, h_prev) + self.bi
        hi = self.fxn.sigmoid(zi)

        # calculates candidate Ct values
        zc = np.dot(self.wcx, ipt) + np.dot(self.wch, h_prev) + self.bc
        hc = np.tanh(zc)

        # calculate new cell state
        ct = (ft * c_prev) + (hi * hc)

        # calculate sigmoid/input component of output
        zo = np.dot(self.wox, ipt) + np.dot(self.woh, h_prev) + self.bo
        ho = self.fxn.sigmoid(zo)

        # calculate output from new cell state and sigmoid component
        ht = ho * np.tanh(ct)

        # z value of output
        zl = np.dot(self.wy, ht) + self.by

        # output following softmax normalization
        y = self.fxn.softmax(zl)

        cache = [ipt, zf, ft, zi, hi, zc, hc, zo, ho, ct, ht, zl, y]
        self.cache.append(cache)
        state = (ht, ct)

        return y, state

    def sample(self, state, seed, rep=200):
        """
        Used for getting a sample output duringi the training process
        :param state: contains the previous output (h_prev) and cell state (c_prev)
        :param seed: a character selected from the text used to seed sequence
        :param rep: the number of characters to generate
        :return: the generated string
        """

        x_t = np.zeros((self.dim, 1))
        x_t[seed] = 1
        ixes = []
        temp_state = state.copy()

        for n in range(rep):
            y, n_state = self.feedforward(x_t, temp_state[-1])
            temp_state.append(n_state)
            ix = np.random.choice(range(self.dim), p=y.ravel())
            x = np.zeros((self.dim, 1))
            x[ix] = 1
            ixes.append(ix)
            x_t = x
        self.cache = []
        return ixes
