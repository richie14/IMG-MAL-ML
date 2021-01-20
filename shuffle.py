import numpy as np
import pandas as pd
import sys

arguments = sys.argv[1]
args = arguments.strip('.csv');

def shuffler(filename):
  df = pd.read_csv(filename, header=0,dtype=object,na_filter=False)
  # return the pandas dataframe
  return df.reindex(np.random.permutation(df.index))


def main(outputfilename):
  shuffler(arguments).to_csv(outputfilename, sep=',',encoding = 'utf-8',index = False)

if __name__ == '__main__': 
  main(args+'-new.csv')
