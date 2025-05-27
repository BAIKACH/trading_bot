import argparse
import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
import os
from dotenv import load_dotenv
from strategies.ma_crossover import MACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.ml_strategy import MLStrategy
from backtesting.backtester import Backtester
from risk_management.position_sizing import calculate_notional_value
from trading.exchanges import place_order, exchanges
import config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка переменных окружения
load_dotenv()

def fetch_coingecko_data(coin_id, days):
    """Получение данных с CoinGecko (заглушка, замените реальным API-вызовом)."""
    # Здесь должен быть вызов API CoinGecko, для примера возвращаем случайные данные
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days * 24, freq='H')
    prices = np.random.normal(30000, 5000, len(dates))  # Примерные цены BTC
    return pd.DataFrame({'price': prices}, index=dates)

def run_bot(strategy, exchange_name, symbol, account_balance):
    """Запуск торгового бота."""
    data = fetch_coingecko_data('bitcoin', days=30)
    signal = strategy.get_signal(data)
    entry_price = data['price'].iloc[-1]
    logging.info(f"Сигнал: {signal}, Цена входа: {entry_price}")

    if signal in ['long', 'short']:
        sl, tp = strategy.get_sl_tp(data, signal)
        notional_value = calculate_notional_value(account_balance, 2, entry_price, sl, data)
        amount = notional_value / entry_price
        side = 'buy' if signal == 'long' else 'sell'
        logging.info(f"Размер позиции: {amount}, SL: {sl}, TP: {tp}")

        # Размещение ордера
        order = place_order(exchange_name, symbol, side, amount)
        logging.info(f"Ордер размещен: {order}")

    # Бэктестинг
    backtester = Backtester(strategy, data)
    results = backtester.run()
    logging.info(f"Результаты бэктестинга:\n{results.tail()}")

def main():
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--start', action='store_true', help='Запустить бота')
    parser.add_argument('--strategy', type=str, default='ma_crossover', help='Выбрать стратегию: ma_crossover, ml, rsi')
    parser.add_argument('--exchange', type=str, default='mexc', help='Выбрать биржу: mexc, binance')
    args = parser.parse_args()

    if args.start:
        account_balance = 1000  # Пример баланса в USDT
        symbol = 'BTC/USDT:USDT'

        if args.strategy == 'ma_crossover':
            strategy = MACrossoverStrategy()
        elif args.strategy == 'ml':
            strategy = MLStrategy(input_size=10, hidden_size=20, output_size=3)
            # Пример обучения ML-стратегии
            data = fetch_coingecko_data('bitcoin', days=30)
            features = data['price'].pct_change().dropna().values.reshape(-1, 1)
            labels = (features > 0).astype(int).flatten()
            strategy.train(features, labels)
        elif args.strategy == 'rsi':
            strategy = RSIStrategy()
        else:
            logging.error("Неизвестная стратегия")
            exit(1)

        run_bot(strategy, args.exchange, symbol, account_balance)
    else:
        logging.info("Используйте --start для запуска бота")

if __name__ == '__main__':
    main()