'''' 
Author: Ramine Tinati: raminetinati AT gmail dot com
'''
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

'''
A simple class to handle different data operations
'''
class DataWrangler:
    wrangler = 1.0

    def __init__(self):
        print('Data Wrangling Tooling')

    def load_data_to_dataframe(self, pathAndFilename=None):
        if pathAndFilename == None:
            print()
        else:
            try:
                dataframe = pd.read_csv(pathAndFilename)
                return dataframe
            except Exception as e:
                print('Error Finding your file')
                print(e)
                return

    def describe_dataframe(self, df, stats=False, histgraphs=False, scattergraph=False):

        print('**Basic Dataframe Details**')
        print('---')
        print('Data Shape:')
        print('Total Records {}'.format(df.shape[0]))
        print('Total Columns {}'.format(df.shape[1]))
        print('Column Names {}'.format(list(df.columns)))
        print('---')
        print('Descriptive Stats:')
        print(df.describe())
        rw, coln = 0, 0
        fig, axs = plt.subplots((df.shape[1]//2)+1, 2, figsize=(20, 20))
        sns.despine(left=True)
        if stats:
            for col in df.columns:
                print('Column: {}'.format(col), end = '. ')
                try:
                    print('Skew/Kurt: {} / {}'.format(df[col].skew(), df[col].kurt()))
                    # print('Kurtosis: {}'.format())
                    if histgraphs:
                        sns.distplot(df[col], ax=axs[rw, coln], label="Distribution of {}".format(col))
                        # print(rw, coln)
                        # plt.title(")
                        # sns.despine()
                        if coln == 0:
                            coln = 1
                        else:
                            rw += 1
                            coln = 0
                except Exception as e:
                    print('Column is non-integer')
                    print(e)
            # plt.setp(axs)
            # plt.tight_layout()
            plt.show()

        if scattergraph:
            print('Cross Column Plot:')
            scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde')
            plt.show()

    def trim_axs(self, axs, N):
        """little helper to massage the axs list to have correct length..."""
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]

    def correlation_analysis(self,df, target_col='Class', corr_threshold = 0.9):

        colinear_feats = []
        df_features = df.drop(target_col, axis=1)
        corr = df_features.corr()
        #explore correlations
    #     print(corr)
        feats = df_features.columns
        for idx, row in corr.iterrows():
    #         print(idx)
            for feat in feats:
    #             print(feat,row[feat])
                if feat != idx:
                    if row[feat] >= corr_threshold:
                        print(idx,feat)
                        colinear_feats.append((idx,feat))

        if len(colinear_feats) == 0:
            print('No Co-Linear Featurs Found, However there may be correlated features depending on their poly transform')
    #     print(df_features.columns)
        f, ax = plt.subplots(figsize=(20, 16))
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax)
        plt.show()
        return colinear_feats