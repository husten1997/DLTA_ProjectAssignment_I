import pandas as pd
import sys as sys
import matplotlib.pyplot as plt
import numpy as np
import itertools as itt
import seaborn as sb

class Descriptive_Analysis:

    data = pd.DataFrame()
    data_details = pd.DataFrame()

    def __init__(self, all_data, all_data_details, coin_ids):

        self.coin_ids = coin_ids
        self.all_data = all_data
        self.all_data_details = all_data_details

        self.setupData(all_data, all_data_details, coin_ids)


    def setupData(self, all_data, all_data_details, coin_ids):

       all_coins = all_data_details.Asset_ID.tolist()

       for k in range(len(self.coin_ids)):

           if self.coin_ids[k] not in all_coins:
               print('Coin ID [' + str(self.coin_ids[k]) + '] is not allowed or does not exist!')
               sys.exit(400)

           self.data = self.data.append(self.all_data[self.all_data.Asset_ID == self.coin_ids[k]])
           self.data_details = self.data_details.append(self.all_data_details[self.all_data_details.Asset_ID == self.coin_ids[k]])

       self.data.sort_values('timestamp', inplace=True)
       self.data_details.sort_values('Asset_ID', inplace=True)
       self.data.reset_index(drop=True, inplace=True)
       self.data_details.reset_index(drop=True, inplace=True)

    #Help function to create subplots
    def computeRows(self):

        tot = len(self.data_details.Asset_ID)
        rows = tot // 2
        rows += tot % 2

        return rows

    #Help function to create subplots
    def createPositionIndex(self):

        return range(1, len(self.data_details.Asset_ID) + 1)

    #Help function to create subplots
    def createFigure(self):

        fig = plt.figure(1)
        fig.set_figheight(20)
        fig.set_figwidth(20)

        return fig

    def createSubplotsReturnOverTime(self):

        rows = self.computeRows()
        position = self.createPositionIndex()
        fig = self.createFigure()

        for k in range(len(self.data_details.Asset_ID)):
            tmp_df = self.data[self.data.Asset_ID == self.data_details.Asset_ID[k]]
            ax = fig.add_subplot(rows, 2, position[k])
            ax.plot(tmp_df.Time, tmp_df.Target)
            ax.set_xlabel('Time', fontsize=15)
            ax.set_ylabel('Target / Return', fontsize=15)
            ax.set_title(self.data_details.Asset_Name[k], fontsize=20)

        plt.show()
        del tmp_df

    def createOnePlotReturnOverTime(self):

        plt.figure(figsize=(12, 8))

        coin_names = self.data_details.Asset_Name.tolist()
        plt.title('Return over time of: ' + ', '.join(coin_names))

        for k in range(len(self.data_details.Asset_ID)):
            tmp_df = self.data[self.data.Asset_ID == self.data_details.Asset_ID[k]]
            plt.plot(tmp_df.Time, tmp_df.Target)

        plt.show()
        del tmp_df

    def createSubplotsReturnDistribution(self):

        rows = self.computeRows()
        position = self.createPositionIndex()
        fig = self.createFigure()

        for k in range(len(self.data_details.Asset_ID)):
            tmp_df = self.data[self.data.Asset_ID == self.data_details.Asset_ID[k]]
            ax = fig.add_subplot(rows, 2, position[k])
            ax.hist(tmp_df.Target, bins=50)
            ax.set_xlim(-0.05, 0.05)
            ax.set_xlabel('Target / Return', fontsize=15)
            ax.set_ylabel('Frequency', fontsize=15)
            ax.set_title(self.data_details.Asset_Name[k], fontsize=20)

        plt.show()
        del tmp_df

    #Help function to create correlation plots
    def prepareDataFrameforCorrelationPlots(self):

        data_btc = self.all_data[self.all_data.Asset_ID == 1]
        data_details_btc = self.all_data_details[self.all_data_details.Asset_ID == 1]

        tmp_df1 = self.data.append(data_btc)
        tmp_df1.sort_values('timestamp', inplace=True)

        tmp_df2 = self.data_details.append(data_details_btc)
        tmp_df2.reset_index(drop=True, inplace=True)

        all_timestamps = np.sort(tmp_df1['timestamp'].unique())
        targets = pd.DataFrame(index=all_timestamps)

        for i, id_ in enumerate(tmp_df2.Asset_ID):
            asset = tmp_df1[tmp_df1.Asset_ID == id_].set_index(keys='timestamp')
            price = pd.Series(index=all_timestamps, data=asset['Close'])
            targets[tmp_df2.Asset_Name[i]] = (
                                                     price.shift(periods=-16) /
                                                     price.shift(periods=-1)
                                             ) - 1

        return targets

    def createSubplotsSevenDayCorrelation(self, df):

        tot = len(df.columns)
        rows = tot // 2
        rows += tot % 2

        position = range(1, tot + 1)

        fig = plt.figure(1)
        fig.set_figheight(20)
        fig.set_figwidth(20)

        step = 0

        for k, j in itt.combinations(df.columns.tolist(), 2):
            corr_time = df.groupby(df.index // (10000 * 60)).corr().loc[:, k].loc[:, j]

            ax = fig.add_subplot(rows, 2, position[step])
            step += 1

            ax.plot(corr_time)
            ax.set_title('7-Days-Corr. between ' + k + ' and ' + j)
            plt.xticks([])
            plt.xlabel("Time")
            plt.ylabel("Correlation")

        plt.show()

    def createHeatMap(self, df):

        sb.heatmap(df.corr())
        plt.show()

