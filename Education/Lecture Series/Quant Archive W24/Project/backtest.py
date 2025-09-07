import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib as plt

# class to contain testing and visualization functions
class backtest:

    trades = []
    portfolioVal = []

    def __init__(self) -> None:
        pass

    # function should step through the algorithm time step by time
    # step and record relevant information, including trades and the
    # value of the portfolio at each step
    def run(algo, start, end):
        pass

    def graphReturns():
        pass

    def calculateVol():
        pass

    def calculateSharpe():
        pass