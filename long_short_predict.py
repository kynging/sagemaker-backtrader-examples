
import backtrader as bt
import json
import math
import numpy as np
import pandas as pd
import talib as ta
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model


class MyStrategy(bt.Strategy):
    
    params=(('printlog', True),)

    def __init__(self):
        super(MyStrategy, self).__init__()
        
        directory = '/root/sagemaker-backtrader-examples'
        image_tag = '3_strategy'
        
        with open("{}/{}/input/config/hyperparameters.json".format(directory, image_tag)) as json_file:
            self.config = json.load(json_file)
        self.config["long_threshold"]=float(self.config["long_threshold"])
        self.config["short_threshold"]=float(self.config["short_threshold"])
        self.config["size"]=int(self.config["size"])
        self.config["profit_target_pct"]=float(self.config["profit_target_pct"])
        self.config["stop_target_pct"]=float(self.config["stop_target_pct"])

        self.order=None
        self.orderPlaced=False
                                
        self.model = load_model('{}/{}/model.h5'.format(directory, image_tag))
        
        # input / indicators
        self.repeat_count = 15
        self.repeat_step = 1
        
        self.profitTarget=self.config["profit_target_pct"]/100.0
        self.stopTarget=self.config["stop_target_pct"]/100.0
        self.size=self.config["size"]
    
        self.sma=[]
        self.roc=[]
        for i in range(0, self.repeat_count):
            self.sma.append(bt.talib.SMA(self.data, timeperiod=(i+1)*self.repeat_step + 1, plot=False))
            self.roc.append(bt.talib.ROC(self.data, timeperiod=(i+1)*self.repeat_step + 1, plot=False))
        
    def next(self):
        super(MyStrategy, self).next()
        
        idx_0 = self.datas[0].datetime.datetime(0)
        close_price = self.datas[0].close
        temp = []
        
        temp2 = []
        temp2.append(close_price)

        ## sma
        for i in range(0, self.repeat_count):
            if math.isnan(self.sma[i][0]):
                temp2.append(close_price)
            else:
                temp2.append(self.sma[i][0])

        min_value = min(temp2)
        max_value = max(temp2)
        for i in temp2:
            if max_value == min_value:
                temp.append(0)
            else:
                temp.append((i - min_value) / (max_value - min_value))

        ## roc
        for i in range(0, self.repeat_count):
            if math.isnan(self.roc[i][0]):
                temp.append(0)
            else:
                temp.append(self.roc[i][0])
        
        ## dataX
        dataX = np.array([np.array(temp)])

        ## dataY
        dataY = self.model.predict(dataX)
        
#         print(len(dataX[0]), len(dataY[0]))
        
        ## 开仓条件
        tLong = dataY[0][0]
        tShort = dataY[0][1]
        if not self.position:
            fLong = (tLong > self.config["long_threshold"]) 
            fShort = (tShort > self.config["short_threshold"])
            if fLong:
                self.order = self.buy(size=self.size)
                self.limitPrice = close_price + self.profitTarget*close_price
                self.stopPrice = close_price - self.stopTarget*close_price
            elif fShort:
                self.order = self.sell(size=self.size)                    
                self.limitPrice = close_price - self.profitTarget*close_price
                self.stopPrice = close_price + self.stopTarget*close_price

        ## 平仓逻辑
        if self.position:
            if self.position.size > 0:
                if close_price >= self.limitPrice or close_price <= self.stopPrice:
                    self.order = self.sell(size=self.size)
            elif self.position.size < 0:
                if close_price <= self.limitPrice or close_price >= self.stopPrice:
                    self.order = self.buy(size=self.size)
                    
    ## 日志记录
    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()},{txt}')

    # 记录交易执行情况（可选，默认不输出结果）
    def notify_order(self, order):
        # 如果 order 为 submitted/accepted，返回空
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 如果 order 为 buy/sell executed，报告价格结果
        if order.status in [order.Completed]: 
            if order.isbuy():
                self.log(f'买入：\n价格：%.2f,\
                交易金额：-%.2f,\
                手续费：%.2f' % (order.executed.price, order.executed.value, order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'卖出:\n价格：%.2f,\
                交易金额：%.2f,\
                手续费：%.2f' % (order.executed.price, order.executed.price*self.size, order.executed.comm))
            self.bar_executed = len(self) 

        # 如果指令取消/交易失败, 报告结果
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('交易失败')
        self.order = None

    # 记录交易收益情况（可省略，默认不输出结果）
    def notify_trade(self,trade):
        if not trade.isclosed:
            return
        self.log(f'策略收益：\n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}')

    # 回测结束后输出结果（可省略，默认输出结果）
    def stop(self):
        self.log('期末总资金 %.2f' %
                 (self.broker.getvalue()), doprint=True)
