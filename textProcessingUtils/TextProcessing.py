from __future__ import division
import os
import sys

# Path for spark source folder
os.environ['SPARK_HOME']="/usr/local/spark"

# Append pyspark  to Python Path
sys.path.append("/usr/local/spark/python")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.streaming import StreamingContext
    from pyspark.ml.feature import NGram
    from pyspark.sql import SQLContext
    from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
    from pyspark.mllib.regression import LabeledPoint
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

def split_chars(x):
    x = str(x.encode('utf-8'))
    y = ''.join(e for e in x if e.isalnum())
    n = list(y)
    return n

#create rdd of senteces from file rdd
def get_sentences(file_rdd):
    sents = file_rdd.glom().map(lambda x: " ".join(x)).flatMap(lambda x: x.split("."))
    return sents

#create bigrams from senteces rdd
def create_bigram(sents):
    bigrams = sents.map(lambda x:split_chars(x))
    bigrams = bigrams.flatMap(lambda x: [((x[i],x[i+1]),1) for i in range(0,len(x)-1)])
    return bigrams

#compute frequency of each bigram in text
def bigram_freq(bigrams):
    freq_bigrams = bigrams.reduceByKey(lambda x,y:x+y) \
    .map(lambda x:(x[1],x[0])) \
    .sortByKey(False)
    switch = freq_bigrams.map(lambda (x,y): (y,x))
    sum_df = sum(switch)
    switch = switch.map(lambda (x,y): (x, y/sum_df))
    return switch

def main():
    sc = SparkContext('local')
###############################################################
    lines = sc.textFile("english.txt")
    sents = get_sentences(lines)
    bigrams = create_bigram(sents) #the ngram percentage rdd (sorted)
    bigrams_freq = bigram_freq(bigrams)
    global_rank = bigrams_freq.zipWithIndex()
    global_rank = global_rank.map(lambda (x,y) : (x[0],y))
    print(global_rank.take(15))