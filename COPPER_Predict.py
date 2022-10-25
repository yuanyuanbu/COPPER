from keras.layers import  Dense, Convolution1D, Dropout, Input, Activation, Flatten,MaxPool1D,add, AveragePooling1D, Bidirectional,GRU,LSTM,Multiply, MaxPooling1D,TimeDistributed,AvgPool1D
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.backend import sigmoid
import math
from sklearn import metrics
import numpy as np
from keras.utils.generic_utils import get_custom_objects
from keras_self_attention import SeqSelfAttention
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
import os,sys,re
from Bio import SeqIO
import argparse
def getdata(datapath):
    if os.path.exists(datapath) == False:
        print('Error: file " %s " does not exist.' % datapath)
        sys.exit(1)
    with open(datapath) as f:
        record = f.readlines()
    if re.search('>', record[0]) == None:
        print('Error: the input file " %s " must be fasta format!' % datapath)
        sys.exit(1)

    sequences = list(SeqIO.parse(datapath, "fasta"))
    sequence = []
    for i in range(len(sequences)):
        sequence.append(str(sequences[i].seq))
    return sequence

def swish(x, beta=1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': swish})

def wordcut(trainsequence):
    b = []

    for index, key in enumerate(trainsequence):
        a = {}
        for inde, key1 in enumerate(key):
            a[inde] = key1
        b.append(a)
    return b
def wordcut2mer(trainsequence):
    b = []

    for index, key in enumerate(trainsequence):
        a = []
        c = len(key)
        for i in range(c-1):
            a.append((key[i] + key[i + 1]))
        b.append(a)
    return b
def wordcut3mer(trainsequence):
    b = []

    for index, key in enumerate(trainsequence):
        a = []
        c = len(key)
        for i in range(c-2):
            a.append((key[i] + key[i + 1]+key[i + 2]))
        b.append(a)
    return b

# dropout activate
def bn_activation_dropout(input):
    input_bn = BatchNormalization(axis=-1)(input)
    input_at = Activation('swish')(input_bn)
    input_dp = Dropout(0.4)(input_at)
    return input_dp

# one-dimensional convoiution
def ConvolutionBlock(input, f, k):
    A1 = Convolution1D(filters=f, kernel_size=k, padding='same')(input)
    A1 = bn_activation_dropout(A1)
    return A1

def InceptionA(input):
    A = ConvolutionBlock(input, 64, 1)
    B = ConvolutionBlock(input, 64, 1)
    B = ConvolutionBlock(B, 64, 5)
    C = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(C, 64, 7)
    C = ConvolutionBlock(C, 64, 7)
    return Concatenate(axis=-1)([A, B, C])

def MultiScale(input):
    A = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(C, 64, 3)
    D = ConvolutionBlock(input, 64, 1)
    D = ConvolutionBlock(D, 64, 5)
    D = ConvolutionBlock(D, 64, 5)
    merge = Concatenate(axis=-1)([A, C, D])
    shortcut_y = Convolution1D(filters=192, kernel_size=1, padding='same')(input)
    shortcut_y = BatchNormalization()(shortcut_y)
    result = add([shortcut_y, merge])
    result = Activation('swish')(result)
    return result
#Models
def createModel1():
    word_input = Input(shape=(29,50), name='word_input')

    word1 = Convolution1D(filters=32, kernel_size=1, padding='same')(word_input)
    overallResult1 = MultiScale(word1)
    overallResult1 =Bidirectional(LSTM(32, return_sequences=True))(overallResult1)
    overallResult1 = SeqSelfAttention(
        attention_activation='swish',
        name='Attention',
    )(overallResult1)

    overallResult1 = Flatten()(overallResult1)
    overallResult = Dense(32, activation='swish')(overallResult1)
    ss_output = Dense(1, activation='sigmoid', name='ss_output')(overallResult)
    return Model(inputs=[word_input], outputs=[ss_output])
def createModel2():
    word_input = Input(shape=(28,90), name='word_input')

    word1 = Convolution1D(filters=32, kernel_size=1, padding='same')(word_input)
    overallResult1 = MultiScale(word1)
    overallResult1 =Bidirectional(LSTM(32, return_sequences=True))(overallResult1)
    overallResult1 = SeqSelfAttention(
        attention_activation='swish',
        name='Attention',
    )(overallResult1)

    overallResult1 = Flatten()(overallResult1)
    overallResult = Dense(32, activation='swish')(overallResult1)
    ss_output = Dense(1, activation='sigmoid', name='ss_output')(overallResult)
    return Model(inputs=[word_input], outputs=[ss_output])
def createModel3():
    word_input = Input(shape=(28,70), name='word_input')

    word1 = Convolution1D(filters=32, kernel_size=1, padding='same')(word_input)
    overallResult1 = MultiScale(word1)
    overallResult1 =Bidirectional(LSTM(32, return_sequences=True))(overallResult1)
    overallResult1 = SeqSelfAttention(
        attention_activation='swish',
        name='Attention',
    )(overallResult1)

    overallResult1 = Flatten()(overallResult1)
    overallResult = Dense(32, activation='swish')(overallResult1)
    ss_output = Dense(1, activation='sigmoid', name='ss_output')(overallResult)
    return Model(inputs=[word_input], outputs=[ss_output])
def createModel4():
    word_input = Input(shape=(29,100), name='word_input')

    word1 = Convolution1D(filters=32, kernel_size=1, padding='same')(word_input)
    overallResult1 = MultiScale(word1)
    overallResult1 =Bidirectional(LSTM(32, return_sequences=True))(overallResult1)
    overallResult1 = SeqSelfAttention(
        attention_activation='swish',
        name='Attention',
    )(overallResult1)

    overallResult1 = Flatten()(overallResult1)
    overallResult = Dense(32, activation='swish')(overallResult1)
    ss_output = Dense(1, activation='sigmoid', name='ss_output')(overallResult)
    return Model(inputs=[word_input], outputs=[ss_output])
def createModel5():
    word_input = Input(shape=(29,100), name='word_input')

    word1 = Convolution1D(filters=32, kernel_size=1, padding='same')(word_input)
    overallResult1 = MultiScale(word1)
    overallResult1 =Bidirectional(LSTM(32, return_sequences=True))(overallResult1)
    overallResult1 = SeqSelfAttention(
        attention_activation='swish',
        name='Attention',
    )(overallResult1)

    overallResult1 = Flatten()(overallResult1)
    overallResult = Dense(32, activation='swish')(overallResult1)
    ss_output = Dense(1, activation='sigmoid', name='ss_output')(overallResult)
    return Model(inputs=[word_input], outputs=[ss_output])
#evaluation
def Twoclassfy_evalu(y_test, y_predict):
    TP = 0                                                                                                                                                                              +0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    for i in range(len(y_test)):
        if y_predict[i] > 0.5 and y_test[i] == 1:
            TP += 1
        if y_predict[i] > 0.5 and y_test[i] == 0:
            FP += 1
            FP_index.append(i)
        if y_predict[i] < 0.5 and y_test[i] == 1:
            FN += 1
            FN_index.append(i)
        if y_predict[i] < 0.5 and y_test[i] == 0:
            TN += 1
    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TN + FN) * (FP + TN) * (TP + FN) * (TP + FP))
    Acc = (TP + TN) / (TP + FP + TN + FN)
    fpr,tpr,thresholds = metrics.roc_curve(y_test,y_predict,pos_label=1)#poslabel正样本的标签
    auc = metrics.auc(fpr,tpr)

    return Sn, Sp, Acc, MCC,auc


def main():
    parser = argparse.ArgumentParser(description='COPPER: A stacking ensemble deep-learning networks for computational recognition exclusive virus-derived small interfering RNAs in plants')
    parser.add_argument('--input', dest='inputpath', type=str, required=True,
                        help='query PVsiRNA sequences to be predicted in fasta format.')
    parser.add_argument('--output', dest='outputfile', type=str, required=False,
                        help='save the prediction results in txt format.')
    args = parser.parse_args()

    inputpath = args.inputpath
    outputfile = args.outputfile
    outputfile_original = outputfile
    if outputfile_original == None:
        outputfile_original = ''
    try:
        #data collect
        t_data = inputpath
        ttrainsequence = getdata(t_data)
        length = len(ttrainsequence)
        ttrainsequence1 = wordcut(ttrainsequence)
        #data1
        data1 = wordcut2mer(ttrainsequence1)
        wv2 = 'word2mer50_4.model'
        wvz = Word2Vec.load(wv2)
        x1 = []
        for i in data1:
            c = []
            a = wvz.wv[i]
            a = np.array(a)
            embedding_matrix = np.zeros((29, 50))
            for i in range(a.shape[0]):
                embedding_matrix[i] = a[i]
            c.append(embedding_matrix)
            x1.append(c)

        x_val1 = np.array(x1)
        x_val1= np.reshape(x_val1,(int(length),29,50))

        # data2
        data2 = wordcut3mer(ttrainsequence1)
        wv2 = 'word3mer90_4.model'
        wvz = Word2Vec.load(wv2)
        x2 = []
        for i in data2:
            c = []
            a = wvz.wv[i]
            a = np.array(a)
            embedding_matrix = np.zeros((28, 90))
            for i in range(a.shape[0]):
                embedding_matrix[i] = a[i]
            c.append(embedding_matrix)
            x2.append(c)
        x_val2 = np.array(x2)
        x_val2 = np.reshape(x_val2, (int(length), 28, 90))

        # data3
        data3 = wordcut3mer(ttrainsequence1)
        wv2 = 'word3mer70_2.model'
        wvz = Word2Vec.load(wv2)
        x3 = []
        for i in data3:
            c = []
            a = wvz.wv[i]
            a = np.array(a)
            embedding_matrix = np.zeros((28, 70))
            for i in range(a.shape[0]):
                embedding_matrix[i] = a[i]
            c.append(embedding_matrix)
            x3.append(c)
        x_val3 = np.array(x3)
        x_val3 = np.reshape(x_val3, (int(length), 28, 70))

        # data4
        data4 = wordcut2mer(ttrainsequence1)
        wv2 = 'fast2mer100_2.model'
        wvz = FastText.load(wv2)
        x4 = []
        for i in data4:
            c = []
            a = wvz.wv[i]
            a = np.array(a)
            embedding_matrix = np.zeros((29, 100))
            for i in range(a.shape[0]):
                embedding_matrix[i] = a[i]
            c.append(embedding_matrix)
            x4.append(c)
        word4 = np.array(x4)
        x_val4 = np.reshape(word4, (int(length), 29, 100))

        # data5
        data5 = wordcut2mer(ttrainsequence1)
        wv2 = 'fast2mer100_4.model'
        wvz = FastText.load(wv2)
        x5 = []
        for i in data5:
            c = []
            a = wvz.wv[i]
            a = np.array(a)
            embedding_matrix = np.zeros((29, 100))
            for i in range(a.shape[0]):
                embedding_matrix[i] = a[i]
            c.append(embedding_matrix)
            x5.append(c)
        word5 = np.array(x5)
        x_val5 = np.reshape(word5, (int(length), 29, 100))

        # load model
        model1 = createModel1()
        model1.load_weights("w2mer50_4.h5")
        y_predict1 = model1.predict({'word_input': x_val1, })

        model2 = createModel2()
        model2.load_weights("w3mer90_4.h5")
        y_predict2 = model2.predict({'word_input': x_val2, })

        model3 = createModel3()
        model3.load_weights("w3mer70_2.h5")
        y_predict3 = model3.predict({'word_input': x_val3, })

        model4 = createModel4()
        model4.load_weights("f2mer100_2.h5")
        y_predict4 = model4.predict({'word_input': x_val4, })

        model5 = createModel5()
        model5.load_weights("f2mer100_4.h5")
        y_predict5 = model5.predict({'word_input': x_val5, })

        import joblib
        from sklearn.linear_model import LogisticRegression
        # model = LogisticRegression()
        # evaluate loaded model on test data
        x_test = np.concatenate((y_predict1, y_predict2, y_predict3, y_predict4, y_predict5), axis=-1)
        model = joblib.load('logistic.pkl')
        predictions = model.predict_proba(x_test)
        sequence = getdata(inputpath)
        seq = []
        for i in sequence:
            seq.append(str(i))
        probability = ['%.5f' % float(i) for i in predictions[:, 1]]
        with open(outputfile, 'w') as f:
            for i in range(int(len(x_test))):
                if float(probability[i]) > 0.5:
                    f.write(probability[i] + '*' + '\t')
                    f.write(seq[i] + '*' + '\t')
                    f.write('1' + '\n')
                else:
                    f.write(probability[i] + '*' + '\t')
                    f.write(seq[i] + '*' + '\t')
                    f.write('0' + '\n')
        print(
            'output are saved in ' + outputfile + ', and those with probability greater than 0.5 are marked with *')




    except Exception as e:
        print('Please check the format of your predicting data!')
        sys.exit(1)



if __name__ == "__main__":
    main()

