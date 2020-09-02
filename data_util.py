import boto3
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as lines
import matplotlib.patches as patches
import pandas as pd
import time


# 从 Athena 中调取数据的函数
def get_query_result(QueryString):
 
    athena = boto3.client('athena')
    s3 = boto3.client('s3')

    def execute_query(QueryString):

        ResultConfiguration = dict([])
        ResultConfiguration['OutputLocation'] = 's3://athena-output-cache/'
        response = athena.start_query_execution(QueryString=QueryString, ResultConfiguration=ResultConfiguration)
        QueryExecutionId = response['QueryExecutionId']

        flag = True
        while flag:
            response = athena.get_query_execution(QueryExecutionId=QueryExecutionId)
            if response['QueryExecution']['Status']['State'] == 'SUCCEEDED':
                flag = False
                OutputLocation = response['QueryExecution']['ResultConfiguration']['OutputLocation']
                return OutputLocation
            elif response['QueryExecution']['Status']['State'] == 'FAILED':
                flag = False
                print(response['QueryExecution']['Status'])
                return
            else:
                time.sleep(0.5)

    output_location = execute_query(QueryString)
#     print(output_location)
    response = s3.get_object(Bucket='athena-output-cache', Key=output_location.split('/')[-1])
    df = pd.read_csv(response['Body'])
    
    return df


# 画 K 线图的函数
def plot_candle_stick(prices):

    n = len(prices)
    
    fig = plt.figure(figsize=(20, 12))

    ax = fig.add_axes([0.1, 0.15, 0.8, 0.7])
    ax.set_facecolor('black')
    ax.set_axisbelow(True)

    ax.grid(False, axis='x')
    ax.grid(True, axis='y')
    ax.set_xlim(-1, n)
    ax.set_ylim(min(prices['low']) * 0.97, max(prices['high']) * 1.03)
    ax.set_xticks(range(0, n, max(int(n / 10), 1)))
    ax.set_xticklabels([prices.index.tolist()[index] for index in ax.get_xticks()])

    for i in range(0, n):
        openPrice = prices['open'].iloc[i]
        closePrice = prices['close'].iloc[i]
        highPrice = prices['high'].iloc[i]
        lowPrice = prices['low'].iloc[i]
        if closePrice > openPrice:
            ax.add_patch(
                patches.Rectangle((i - 0.2, openPrice), 0.4, closePrice - openPrice, fill=False, color='r'))
            ax.plot([i, i], [lowPrice, openPrice], 'r')
            ax.plot([i, i], [closePrice, highPrice], 'r')
        else:
            ax.add_patch(patches.Rectangle((i - 0.2, openPrice), 0.4, closePrice - openPrice, color='c'))
            ax.plot([i, i], [lowPrice, highPrice], color='c')
            
    return fig
