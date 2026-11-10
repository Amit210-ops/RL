import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, List, Any
import warnings
import os
import sys
import json
import time
import pickle
from pathlib import Path
from collections import deque

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MIN = 30
TOTAL_SESSION_MINUTES = (MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MIN) - (MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MIN)
BARS_PER_DAY = 25

OVERNIGHT_PENALTY_REWARD = -50.0
HOLDING_COST_PER_BAR = 0.5
MAX_BARS_IN_POSITION = 50

DAILY_LOSS_LIMIT_INR = 5000.0
PER_TRADE_SL_PCT = 0.30
PER_TRADE_TP_PCT = 0.60
MAX_TRADES_PER_DAY = 6
DEFAULT_LOT_SIZE = 65

OBSERVATION_FEATURES = [
    'spot_price', 'open', 'high', 'low',
    'ce_mid', 'ce_bid', 'ce_ask', 'pe_mid', 'pe_bid', 'pe_ask',
    'atm_strike', 'ce_intrinsic', 'pe_intrinsic', 'time_value',
    'ce_iv', 'pe_iv', 'iv_spread',
    'sma_10', 'sma_20', 'sma_slope', 'rsi_14',
    'momentum_1', 'momentum_5', 'momentum_25',
    'returns', 'realized_vol', 'atr', 'zscore_20',
    'volume', 'volume_sma_10', 'volume_ratio',
    'ce_delta', 'ce_vega', 'ce_theta', 'ce_gamma',
    'pe_delta', 'pe_vega', 'pe_theta', 'pe_gamma',
    'hour', 'minute', 'day_of_week', 'minutes_into_day',
    'bar_number', 'session_phase', 'bars_to_expiry', 'is_expiry_day',
    'days_ahead', 'hours_to_expiry',
]

PRICE_COLUMNS = [
    'spot_price', 'open', 'high', 'low',
    'ce_mid', 'ce_bid', 'ce_ask', 'pe_mid', 'pe_bid', 'pe_ask',
    'atm_strike', 'ce_intrinsic', 'pe_intrinsic', 'time_value',
    'sma_10', 'sma_20', 'atr',
]


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

class NormalizationParams:
    """
    FIX vs earlier version: fit() must ONLY ever be called on the training
    split, never on the full (train+test) dataset -- otherwise test-period
    statistics leak backward into how training data gets scaled. See
    DataLoader.load_and_prepare_split() below, which splits BEFORE fitting.
    """

    def __init__(self):
        self.price_scale: float = 1.0
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.iv_scale: float = 100.0
        self.volume_log_shift: float = 1.0
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, method: str = "expanding"):
        if method == "expanding":
            self.price_scale = float(df['spot_price'].iloc[-1]) if 'spot_price' in df.columns else 10000.0
        else:
            self.price_scale = float(df['spot_price'].mean()) if 'spot_price' in df.columns else 10000.0

        for col in df.columns:
            if col in df.select_dtypes(include=[np.number]).columns:
                vals = df[col].dropna()
                self.feature_stats[col] = {
                    'mean': float(vals.mean()),
                    'std': float(vals.std()) if vals.std() > 0 else 1.0,
                    'q01': float(vals.quantile(0.01)),
                    'q99': float(vals.quantile(0.99)),
                    'min': float(vals.min()),
                    'max': float(vals.max()),
                }

        self.is_fitted = True
        return self

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'price_scale': self.price_scale,
                'feature_stats': self.feature_stats,
                'iv_scale': self.iv_scale,
                'volume_log_shift': self.volume_log_shift,
            }, f)
        print(f"  Normalization params saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'NormalizationParams':
        params = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        params.price_scale = data['price_scale']
        params.feature_stats = data['feature_stats']
        params.iv_scale = data.get('iv_scale', 100.0)
        params.volume_log_shift = data.get('volume_log_shift', 1.0)
        params.is_fitted = True
        return params

    def normalize_row(self, row: Dict[str, float], current_spot: float = None) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("NormalizationParams not fitted. Call .fit() or .load() first.")

        spot = current_spot if current_spot else self.price_scale
        normalized = {}

        for col, val in row.items():
            if col in PRICE_COLUMNS:
                normalized[col] = val / spot if spot > 0 else 0.0
            elif col in ['ce_iv', 'pe_iv']:
                normalized[col] = val / self.iv_scale
            elif col in ['volume', 'volume_sma_10']:
                normalized[col] = np.log1p(abs(val)) / 15.0
            elif col in ['ce_vega', 'pe_vega', 'ce_theta', 'pe_theta']:
                sign = np.sign(val)
                log_val = sign * np.log1p(abs(val))
                stats = self.feature_stats.get(col, {'q01': -5, 'q99': 5})
                q01, q99 = stats['q01'], stats['q99']
                if q99 > q01:
                    normalized[col] = np.clip((log_val - q01) / (q99 - q01) * 2 - 1, -1, 1)
                else:
                    normalized[col] = 0.0
            elif col == 'rsi_14':
                normalized[col] = val / 100.0
            elif col in ['minutes_into_day']:
                normalized[col] = val / TOTAL_SESSION_MINUTES
            elif col in ['bar_number']:
                normalized[col] = val / BARS_PER_DAY
            elif col in ['day_of_week']:
                normalized[col] = val / 4.0
            elif col in ['hour']:
                normalized[col] = (val - 9) / 6.0
            elif col in ['minute']:
                normalized[col] = val / 60.0
            else:
                stats = self.feature_stats.get(col)
                if stats and stats['q99'] > stats['q01']:
                    clipped = np.clip(val, stats['q01'], stats['q99'])
                    normalized[col] = (clipped - stats['q01']) / (stats['q99'] - stats['q01']) * 2 - 1
                else:
                    normalized[col] = val

        return normalized


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING COST MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class IndiaTradingCosts:
    @staticmethod
    def compute_total_cost(entry_price: float, exit_price: float,
                           lot_size: int, is_intraday: bool = True) -> float:
        if entry_price <= 0 or exit_price <= 0 or lot_size <= 0:
            return 0.0

        buy_value = entry_price * lot_size
        sell_value = exit_price * lot_size
        turnover = buy_value + sell_value

        brokerage = 40.0
        stt = (0.000625 if is_intraday else 0.001) * sell_value
        exchange_txn = 0.000495 * turnover
        sebi = 10.0 / 1e7 * turnover
        stamp = 0.00003 * buy_value
        gst = 0.18 * (brokerage + exchange_txn + sebi)
        ipft = 10.0 / 1e7 * turnover

        return brokerage + stt + exchange_txn + sebi + stamp + gst + ipft

    @staticmethod
    def compute_slippage(price: float, bid: float, ask: float,
                         is_expiry_day: bool = False) -> float:
        if price <= 0:
            return 0.5
        if bid > 0 and ask > 0 and ask > bid:
            base_slip = (ask - bid) * 0.5
        else:
            base_slip = price * 0.005
        if is_expiry_day:
            base_slip *= 1.5
        return max(0.25, base_slip)


# ═══════════════════════════════════════════════════════════════════════════════
# KELLY CRITERION
# ═══════════════════════════════════════════════════════════════════════════════

class KellyCriterion:
    def __init__(self, lookback: int = 50, min_trades: int = 10):
        self.lookback = lookback
        self.min_trades = min_trades
        self.trade_returns: List[float] = []

    def update(self, pnl_pct: float):
        self.trade_returns.append(pnl_pct)
        if len(self.trade_returns) > self.lookback * 2:
            self.trade_returns = self.trade_returns[-self.lookback:]

    def optimal_lots(self, max_lots: int = 3) -> int:
        if len(self.trade_returns) < self.min_trades:
            return 1
        recent = self.trade_returns[-self.lookback:]
        wins = [r for r in recent if r > 0]
        losses = [r for r in recent if r < 0]
        if not wins or not losses:
            return 1
        win_rate = len(wins) / len(recent)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        if avg_loss == 0:
            return max_lots
        b = avg_win / avg_loss
        half_kelly = ((win_rate * b - (1 - win_rate)) / b) * 0.5
        if half_kelly <= 0.1:
            return 1
        elif half_kelly < 0.3:
            return min(2, max_lots)
        else:
            return min(3, max_lots)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADER  (FIXED: split BEFORE fitting normalization)
# ═══════════════════════════════════════════════════════════════════════════════

class DataLoader:

    @staticmethod
    def _load_raw(csv_path: str) -> pd.DataFrame:
        print(f"\n  Loading data from: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        print(f"  Raw shape: {df.shape}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Total bars: {len(df):,}")

        forward_cols = [c for c in df.columns if any(
            c.startswith(prefix) for prefix in
            ['spot_t', 'ce_mid_t', 'pe_mid_t', 'ce_delta_t', 'pe_delta_t',
             'ce_iv_t', 'pe_iv_t', 'spot_chg_t']
        )]
        if forward_cols:
            df.drop(columns=forward_cols, inplace=True)
            print(f"  Dropped {len(forward_cols)} forward-looking columns")

        drop_cols = [c for c in ['expiry'] if c in df.columns]
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"  Filling {nan_count} NaN values")
            df.fillna(method='ffill', inplace=True)
            df.fillna(0.0, inplace=True)

        df = DataLoader._add_derived_features(df)
        return df

    @staticmethod
    def load_and_prepare_split(
        csv_path: str,
        train_ratio: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 'NormalizationParams']:
        """
        Load, split chronologically FIRST, then fit normalization on the
        training split only, then apply it to both splits.
        """
        df = DataLoader._load_raw(csv_path)

        split_idx = int(len(df) * train_ratio)
        raw_train = df.iloc[:split_idx].copy()
        raw_test = df.iloc[split_idx:].copy()
        print(f"  Train: {len(raw_train):,} bars ({raw_train.index.min()} to {raw_train.index.max()})")
        print(f"  Test:  {len(raw_test):,} bars ({raw_test.index.min()} to {raw_test.index.max()})")

        norm_params = NormalizationParams()
        norm_params.fit(raw_train, method="expanding")

        norm_train = DataLoader._normalize_features(raw_train, norm_params, is_training=True)
        norm_test = DataLoader._normalize_features(raw_test, norm_params, is_training=False)

        print(f"  Normalized train shape: {norm_train.shape}")
        print(f"  Normalized test shape:  {norm_test.shape}")

        return raw_train, raw_test, norm_train, norm_test, norm_params

    @staticmethod
    def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        if 'ce_bid' in df.columns and 'ce_ask' in df.columns:
            df['ce_spread'] = df['ce_ask'] - df['ce_bid']
            df['pe_spread'] = df['pe_ask'] - df['pe_bid']
            df['ce_spread_pct'] = df['ce_spread'] / (df['ce_mid'] + 0.01)
            df['pe_spread_pct'] = df['pe_spread'] / (df['pe_mid'] + 0.01)

        if 'pe_mid' in df.columns and 'ce_mid' in df.columns:
            df['pcr'] = df['pe_mid'] / (df['ce_mid'] + 0.01)
            df['pcr_signal'] = df['pcr'] - 1.0

        if 'ce_iv' in df.columns:
            df['iv_rank'] = df['ce_iv'].rolling(100, min_periods=10).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 0.001)
            ).fillna(0.5)

        if 'hours_to_expiry' in df.columns:
            df['theta_urgency'] = 1.0 / (np.sqrt(df['hours_to_expiry'] / 31.25) + 0.01)
            df['theta_urgency'] = df['theta_urgency'].clip(0, 10) / 10.0

        if 'sma_slope' in df.columns:
            df['trend_up'] = (df['sma_slope'] > 0).astype(float)
            df['trend_down'] = (df['sma_slope'] < 0).astype(float)

        if 'rsi_14' in df.columns:
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(float)
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(float)

        if all(c in df.columns for c in ['momentum_1', 'momentum_5']):
            df['mom_aligned_up'] = ((df['momentum_1'] > 0) & (df['momentum_5'] > 0)).astype(float)
            df['mom_aligned_down'] = ((df['momentum_1'] < 0) & (df['momentum_5'] < 0)).astype(float)

        if 'bar_number' in df.columns:
            df['bars_remaining_norm'] = (BARS_PER_DAY - df['bar_number']).clip(0, BARS_PER_DAY) / BARS_PER_DAY

        if 'ce_mid' in df.columns and 'spot_price' in df.columns:
            df['ce_premium_pct'] = df['ce_mid'] / (df['spot_price'] + 1) * 100
            df['pe_premium_pct'] = df['pe_mid'] / (df['spot_price'] + 1) * 100

        return df

    @staticmethod
    def _normalize_features(df: pd.DataFrame, params: 'NormalizationParams',
                           is_training: bool) -> pd.DataFrame:
        norm_df = df.copy()

        if is_training:
            expanding_spot = df['spot_price'].expanding(min_periods=1).mean()
        else:
            expanding_spot = pd.Series(params.price_scale, index=df.index)

        for col in PRICE_COLUMNS:
            if col in norm_df.columns:
                norm_df[col] = norm_df[col] / expanding_spot

        for col in ['ce_iv', 'pe_iv']:
            if col in norm_df.columns:
                norm_df[col] = norm_df[col] / params.iv_scale

        for col in ['volume', 'volume_sma_10']:
            if col in norm_df.columns:
                norm_df[col] = np.log1p(np.abs(norm_df[col])) / 15.0

        for col in ['ce_vega', 'pe_vega', 'ce_theta', 'pe_theta']:
            if col in norm_df.columns:
                sign = np.sign(norm_df[col])
                norm_df[col] = sign * np.log1p(np.abs(norm_df[col]))
                stats = params.feature_stats.get(col)
                if stats and stats['q99'] > stats['q01']:
                    norm_df[col] = norm_df[col].clip(stats['q01'], stats['q99'])
                    norm_df[col] = (norm_df[col] - stats['q01']) / (stats['q99'] - stats['q01']) * 2 - 1

        if 'rsi_14' in norm_df.columns:
            norm_df['rsi_14'] = norm_df['rsi_14'] / 100.0
        if 'minutes_into_day' in norm_df.columns:
            norm_df['minutes_into_day'] = norm_df['minutes_into_day'] / TOTAL_SESSION_MINUTES
        if 'bar_number' in norm_df.columns:
            norm_df['bar_number'] = norm_df['bar_number'] / BARS_PER_DAY
        if 'hour' in norm_df.columns:
            norm_df['hour'] = (norm_df['hour'] - 9) / 6.0
        if 'minute' in norm_df.columns:
            norm_df['minute'] = norm_df['minute'] / 60.0
        if 'day_of_week' in norm_df.columns:
            norm_df['day_of_week'] = norm_df['day_of_week'] / 4.0

        numeric_cols = norm_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            norm_df[col] = norm_df[col].clip(-5, 5)

        norm_df.replace([np.inf, -np.inf], 0.0, inplace=True)
        norm_df.fillna(0.0, inplace=True)

        return norm_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class NiftyCombinedOptionsEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        raw_df: pd.DataFrame,
        norm_df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        lot_size: int = DEFAULT_LOT_SIZE,
        max_lots: int = 3,
        sl_pct: float = PER_TRADE_SL_PCT,
        tp_pct: float = PER_TRADE_TP_PCT,
        daily_loss_limit: float = DAILY_LOSS_LIMIT_INR,
        max_trades_per_day: int = MAX_TRADES_PER_DAY,
        reward_scaling: float = 1.0,
    ):
        super().__init__()

        self.raw_df = raw_df.copy()
        self.norm_df = norm_df.copy()

        assert len(self.raw_df) == len(self.norm_df), "Raw and norm DataFrames must have same length"
        self.max_steps = len(self.raw_df) - 1

        self.initial_capital = float(initial_capital)
        self.lot_size = int(lot_size)
        self.max_lots = int(max_lots)

        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.daily_loss_limit = daily_loss_limit
        self.max_trades_per_day = max_trades_per_day
        self.reward_scaling = reward_scaling

        self.cost_calc = IndiaTradingCosts()
        self.kelly = KellyCriterion()

        all_feature_candidates = OBSERVATION_FEATURES + [
            'ce_spread', 'pe_spread', 'ce_spread_pct', 'pe_spread_pct',
            'pcr', 'pcr_signal', 'iv_rank', 'theta_urgency',
            'trend_up', 'trend_down', 'rsi_oversold', 'rsi_overbought',
            'mom_aligned_up', 'mom_aligned_down',
            'bars_remaining_norm', 'ce_premium_pct', 'pe_premium_pct'
        ]
        self.feature_columns = [c for c in all_feature_candidates if c in self.norm_df.columns]

        self.action_space = spaces.Discrete(5)
        obs_dim = len(self.feature_columns) + 8
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._state_reset()

    def _state_reset(self):
        self.current_step = 0
        self.capital = self.initial_capital
        self.position_type = None
        self.position_entry_price = 0.0
        self.position_entry_step = 0
        self.position_lots = 0
        self.bars_in_position = 0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_trade_date = None
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trade_pnls: List[float] = []
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state_reset()
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        raw_row = self.raw_df.iloc[self.current_step]
        ts = self.raw_df.index[self.current_step]
        reward = 0.0

        ce_mid_raw = float(raw_row.get('ce_mid', 0))
        pe_mid_raw = float(raw_row.get('pe_mid', 0))
        ce_bid_raw = float(raw_row.get('ce_bid', ce_mid_raw * 0.99))
        ce_ask_raw = float(raw_row.get('ce_ask', ce_mid_raw * 1.01))
        pe_bid_raw = float(raw_row.get('pe_bid', pe_mid_raw * 0.99))
        pe_ask_raw = float(raw_row.get('pe_ask', pe_mid_raw * 1.01))
        is_expiry = float(raw_row.get('is_expiry_day', 0)) > 0.5
        bars_to_exp = float(raw_row.get('bars_to_expiry', 99))
        hours_to_exp = float(raw_row.get('hours_to_expiry', 99))

        self._check_new_day(ts)

        if action == 0:
            reward += self._reward_hold(raw_row)
        elif action == 1:
            reward += self._execute_buy('CE', ce_mid_raw, ce_bid_raw, ce_ask_raw, is_expiry, hours_to_exp)
        elif action == 2:
            reward += self._execute_sell('CE', ce_mid_raw, ce_bid_raw, is_expiry)
        elif action == 3:
            reward += self._execute_buy('PE', pe_mid_raw, pe_bid_raw, pe_ask_raw, is_expiry, hours_to_exp)
        elif action == 4:
            reward += self._execute_sell('PE', pe_mid_raw, pe_bid_raw, is_expiry)

        if self.position_type is not None:
            self.bars_in_position += 1
            current_price = ce_mid_raw if self.position_type == 'CE' else pe_mid_raw
            if self.position_entry_price > 0:
                self.unrealized_pnl = (current_price - self.position_entry_price) * self.position_lots * self.lot_size
            reward += self._holding_cost_reward(hours_to_exp, ts)
            reward += self._urgency_reward(current_price, is_expiry, bars_to_exp, hours_to_exp)

        scaled_reward = float(np.clip(reward / 100.0 * self.reward_scaling, -10, 10))

        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        if terminated and self.position_type is not None:
            last_raw = self.raw_df.iloc[-1]
            last_price = float(last_raw.get('ce_mid', 0)) if self.position_type == 'CE' else float(last_raw.get('pe_mid', 0))
            end_pnl = self._force_close(last_price)
            scaled_reward += float(np.clip(end_pnl / 100.0 * self.reward_scaling, -10, 10))

        current_equity = self.capital + self.unrealized_pnl
        self.peak_capital = max(self.peak_capital, current_equity)
        if self.peak_capital > 0:
            dd = (self.peak_capital - current_equity) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, dd)

        return self._get_obs(), scaled_reward, terminated, False, self._get_info()

    def _execute_buy(self, option_type, mid, bid, ask, is_expiry, hours_to_exp):
        if self.position_type is not None:
            return -200.0
        if mid <= 0.5:
            return -100.0
        if self.daily_trades >= self.max_trades_per_day:
            return -50.0
        if self.daily_pnl <= -self.daily_loss_limit:
            return -50.0
        if hours_to_exp < 0.5:
            return -100.0

        lots = self.kelly.optimal_lots(self.max_lots)
        slippage = self.cost_calc.compute_slippage(mid, bid, ask, is_expiry_day=is_expiry)
        entry_price = mid + slippage

        total_premium = entry_price * lots * self.lot_size
        if total_premium > self.capital * 0.5:
            lots = max(1, int(self.capital * 0.5 / (entry_price * self.lot_size)))
            total_premium = entry_price * lots * self.lot_size
        if total_premium > self.capital:
            return -50.0

        self.position_type = option_type
        self.position_entry_price = entry_price
        self.position_entry_step = self.current_step
        self.position_lots = lots
        self.bars_in_position = 0
        self.capital -= total_premium
        self.daily_trades += 1

        return -slippage * lots * self.lot_size

    def _execute_sell(self, option_type, mid, bid, is_expiry):
        if self.position_type is None:
            return -200.0
        if self.position_type != option_type:
            return -200.0
        if self.position_entry_price <= 0:
            self.position_type = None
            return -50.0

        slippage = self.cost_calc.compute_slippage(mid, bid, mid * 1.01, is_expiry_day=is_expiry)
        exit_price = max(0.05, mid - slippage)
        lots = self.position_lots
        gross_pnl = (exit_price - self.position_entry_price) * lots * self.lot_size

        is_intraday = self.bars_in_position < BARS_PER_DAY
        total_cost = self.cost_calc.compute_total_cost(
            self.position_entry_price, exit_price, lots * self.lot_size, is_intraday
        )
        net_pnl = gross_pnl - total_cost

        self.capital += exit_price * lots * self.lot_size
        self.realized_pnl += net_pnl
        self.daily_pnl += net_pnl
        self.total_trades += 1
        self.trade_pnls.append(net_pnl)

        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        pnl_pct = net_pnl / (self.position_entry_price * lots * self.lot_size + 1e-10)
        self.kelly.update(pnl_pct)

        self.position_type = None
        self.position_entry_price = 0.0
        self.position_entry_step = 0
        self.position_lots = 0
        self.bars_in_position = 0
        self.unrealized_pnl = 0.0

        return net_pnl

    def _force_close(self, price):
        if self.position_type is None:
            return 0.0
        exit_price = max(0.05, price)
        lots = self.position_lots
        gross_pnl = (exit_price - self.position_entry_price) * lots * self.lot_size
        total_cost = self.cost_calc.compute_total_cost(self.position_entry_price, exit_price, lots * self.lot_size)
        net_pnl = gross_pnl - total_cost

        self.capital += exit_price * lots * self.lot_size
        self.realized_pnl += net_pnl
        self.total_trades += 1
        self.trade_pnls.append(net_pnl)

        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        self.position_type = None
        self.position_entry_price = 0.0
        self.position_lots = 0
        self.bars_in_position = 0
        self.unrealized_pnl = 0.0

        return net_pnl

    def _reward_hold(self, raw_row):
        if self.position_type is None:
            mom1 = float(raw_row.get('momentum_1', 0))
            if abs(mom1) > 50:
                return -5.0
        return 0.0

    def _holding_cost_reward(self, hours_to_exp, ts):
        cost = 0.0
        if self.current_step > 0:
            prev_ts = self.raw_df.index[self.current_step - 1]
            if isinstance(ts, pd.Timestamp) and isinstance(prev_ts, pd.Timestamp):
                if ts.date() != prev_ts.date():
                    cost += OVERNIGHT_PENALTY_REWARD
        if hours_to_exp < 2:
            cost -= HOLDING_COST_PER_BAR * 10
        elif hours_to_exp < 5:
            cost -= HOLDING_COST_PER_BAR * 3
        else:
            cost -= HOLDING_COST_PER_BAR
        return cost

    def _urgency_reward(self, current_price, is_expiry, bars_to_exp, hours_to_exp):
        urgency = 0.0
        if self.position_entry_price > 0:
            pnl_pct = (current_price - self.position_entry_price) / self.position_entry_price
            if pnl_pct < -self.sl_pct:
                urgency -= 100.0
            elif pnl_pct < -self.sl_pct * 0.7:
                urgency -= 30.0
            if pnl_pct > self.tp_pct:
                urgency -= 20.0
        if is_expiry and bars_to_exp <= 3:
            urgency -= 50.0
        if self.bars_in_position >= MAX_BARS_IN_POSITION:
            urgency -= 80.0
        elif self.bars_in_position >= MAX_BARS_IN_POSITION * 0.7:
            urgency -= 20.0
        return urgency

    def _get_obs(self):
        idx = min(self.current_step, self.max_steps)
        norm_row = self.norm_df.iloc[idx]
        features = [float(norm_row.get(col, 0.0)) for col in self.feature_columns]
        position_state = [
            1.0 if self.position_type == 'CE' else 0.0,
            1.0 if self.position_type == 'PE' else 0.0,
            min(1.0, self.bars_in_position / MAX_BARS_IN_POSITION),
            np.clip(self.realized_pnl / (self.initial_capital * 0.1 + 1e-10), -1, 1),
            np.clip(self.daily_pnl / (self.daily_loss_limit + 1e-10), -1, 1),
            np.clip(self.unrealized_pnl / (self.initial_capital * 0.05 + 1e-10), -1, 1),
            self.daily_trades / (self.max_trades_per_day + 1e-10),
            1.0 if self.position_type is not None else 0.0,
        ]
        obs = np.array(features + position_state, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(obs, -10.0, 10.0)

    def _get_info(self):
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        avg_pnl = np.mean(self.trade_pnls) if self.trade_pnls else 0.0
        return {
            "capital": round(self.capital, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "position_type": self.position_type,
            "position_lots": self.position_lots,
            "bars_in_position": self.bars_in_position,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(win_rate, 1),
            "avg_trade_pnl": round(avg_pnl, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "kelly_lots": self.kelly.optimal_lots(self.max_lots),
            "step": self.current_step,
        }

    def _check_new_day(self, ts):
        if not isinstance(ts, pd.Timestamp):
            return
        current_date = ts.date()
        if self.last_trade_date != current_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_trade_date = current_date


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

class LiveInference:
    ACTION_NAMES = {0: "HOLD", 1: "BUY_CE", 2: "SELL_CE", 3: "BUY_PE", 4: "SELL_PE"}

    def __init__(self, model_path, norm_params_path, feature_columns_path, lot_size=DEFAULT_LOT_SIZE):
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            print(f"  Model loaded: {model_path}")
        except Exception as e:
            print(f"  Model load error: {e}")
            self.model = None

        self.norm_params = NormalizationParams.load(norm_params_path)
        with open(feature_columns_path, 'r') as f:
            self.feature_columns = json.load(f)

        self.lot_size = lot_size
        self.position_type = None
        self.position_entry_price = 0.0
        self.position_lots = 0
        self.bars_in_position = 0
        self.realized_pnl = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.kelly = KellyCriterion()

    def get_action(self, market_data, current_spot=None):
        if self.model is None:
            return {"action": 0, "action_name": "HOLD", "error": "No model loaded"}
        normalized = self.norm_params.normalize_row(market_data, current_spot=current_spot)
        obs = self._build_obs(normalized)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        is_valid, reason = self._validate(action, market_data)
        return {
            "action": action, "action_name": self.ACTION_NAMES[action],
            "is_valid": is_valid, "validation_note": reason,
            "position": self.position_type, "position_lots": self.position_lots,
            "daily_pnl": round(self.daily_pnl, 2), "daily_trades": self.daily_trades,
            "kelly_lots": self.kelly.optimal_lots(),
        }

    def _build_obs(self, normalized_data):
        features = [float(normalized_data.get(col, 0)) for col in self.feature_columns]
        state = [
            1.0 if self.position_type == 'CE' else 0.0,
            1.0 if self.position_type == 'PE' else 0.0,
            min(1.0, self.bars_in_position / MAX_BARS_IN_POSITION),
            np.clip(self.realized_pnl / 10000, -1, 1),
            np.clip(self.daily_pnl / DAILY_LOSS_LIMIT_INR, -1, 1),
            0.0,
            self.daily_trades / MAX_TRADES_PER_DAY,
            1.0 if self.position_type else 0.0,
        ]
        obs = np.array(features + state, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(obs, -10.0, 10.0)

    def _validate(self, action, raw_data):
        if action == 2 and self.position_type != 'CE':
            return False, "SELL_CE but no CE position"
        if action == 4 and self.position_type != 'PE':
            return False, "SELL_PE but no PE position"
        if action in [1, 3] and self.position_type is not None:
            return False, f"BUY but already in {self.position_type}"
        if self.daily_trades >= MAX_TRADES_PER_DAY and action in [1, 3]:
            return False, "Daily trade limit reached"
        hours_exp = float(raw_data.get('hours_to_expiry', 99))
        if hours_exp < 0.5 and action in [1, 3]:
            return False, "Too close to expiry"
        return True, "OK"

    def update_position(self, action, price, lots=None):
        action_name = self.ACTION_NAMES[action]
        if lots is None:
            lots = self.kelly.optimal_lots()
        if action_name in ('BUY_CE', 'BUY_PE'):
            self.position_type = action_name.split('_')[1]
            self.position_entry_price = price
            self.position_lots = lots
            self.bars_in_position = 0
            self.daily_trades += 1
        elif action_name in ('SELL_CE', 'SELL_PE'):
            if self.position_entry_price > 0:
                pnl = (price - self.position_entry_price) * self.position_lots * self.lot_size
                self.realized_pnl += pnl
                self.daily_pnl += pnl
                pnl_pct = pnl / (self.position_entry_price * self.position_lots * self.lot_size + 1e-10)
                self.kelly.update(pnl_pct)
            self.position_type = None
            self.position_entry_price = 0.0
            self.position_lots = 0
            self.bars_in_position = 0
        elif action_name == 'HOLD' and self.position_type:
            self.bars_in_position += 1

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════════

class Backtester:
    ACTION_NAMES = {0: "HOLD", 1: "BUY_CE", 2: "SELL_CE", 3: "BUY_PE", 4: "SELL_PE"}

    def __init__(self, model, env, deterministic=True):
        self.model = model
        self.env = env
        self.deterministic = deterministic

    def run(self):
        obs, info = self.env.reset()
        done = False
        all_actions = []
        equity_curve = [self.env.initial_capital]

        while not done:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            action = int(action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            all_actions.append(action)
            equity_curve.append(info.get("capital", equity_curve[-1]) + info.get("unrealized_pnl", 0))

        equity_curve = np.array(equity_curve)
        trade_pnls = self.env.trade_pnls
        total_pnl = info.get("realized_pnl", 0)

        returns = np.diff(equity_curve) / (np.abs(equity_curve[:-1]) + 1e-10)
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(BARS_PER_DAY * 252)

        peak = np.maximum.accumulate(equity_curve)
        max_dd = np.max((peak - equity_curve) / (peak + 1e-10))

        gross_profit = sum(p for p in trade_pnls if p > 0) if trade_pnls else 0
        gross_loss = abs(sum(p for p in trade_pnls if p < 0)) if trade_pnls else 1
        profit_factor = gross_profit / (gross_loss + 1e-10)

        action_counts = {i: all_actions.count(i) for i in range(5)}

        return {
            "total_pnl_inr": round(total_pnl, 2),
            "total_trades": info.get("total_trades", 0),
            "win_rate_pct": round(info.get("win_rate", 0), 1),
            "profit_factor": round(profit_factor, 3),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_trade_pnl_inr": round(np.mean(trade_pnls), 2) if trade_pnls else 0,
            "best_trade_inr": round(max(trade_pnls), 2) if trade_pnls else 0,
            "worst_trade_inr": round(min(trade_pnls), 2) if trade_pnls else 0,
            "action_distribution": action_counts,
            "total_bars": len(all_actions),
            "final_capital": round(equity_curve[-1], 2),
            "return_pct": round((equity_curve[-1] / equity_curve[0] - 1) * 100, 2),
        }

    def print_report(self, results):
        print("\n" + "=" * 70)
        print("  BACKTEST REPORT - NIFTY OPTIONS (CE + PE BUYING)")
        print("=" * 70)
        print(f"  Total PnL:      Rs.{results['total_pnl_inr']:,.2f}")
        print(f"  Return:         {results['return_pct']}%")
        print(f"  Final Capital:  Rs.{results['final_capital']:,.2f}")
        print(f"  Total Trades:   {results['total_trades']}")
        print(f"  Win Rate:       {results['win_rate_pct']}%")
        print(f"  Profit Factor:  {results['profit_factor']}")
        print(f"  Sharpe Ratio:   {results['sharpe_ratio']}")
        print(f"  Max Drawdown:   {results['max_drawdown_pct']}%")
        print(f"  Avg Trade PnL:  Rs.{results['avg_trade_pnl_inr']:,.2f}")
        print(f"  Best Trade:     Rs.{results['best_trade_inr']:,.2f}")
        print(f"  Worst Trade:    Rs.{results['worst_trade_inr']:,.2f}")
        print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from stable_baselines3.common.callbacks import BaseCallback

    class TradingMetricsCallback(BaseCallback):
        def __init__(self, log_freq=2000, verbose=0):
            super().__init__(verbose)
            self.log_freq = log_freq

        def _on_step(self):
            if self.n_calls % self.log_freq == 0:
                infos = self.locals.get("infos", [])
                for info in infos:
                    if "total_trades" in info:
                        self.logger.record("trading/trades", info["total_trades"])
                        self.logger.record("trading/win_rate", info.get("win_rate", 0))
                        self.logger.record("trading/realized_pnl", info.get("realized_pnl", 0))
                        self.logger.record("trading/max_dd", info.get("max_drawdown_pct", 0))
                        break
            return True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN  (FIXED: split before fit)
# ═══════════════════════════════════════════════════════════════════════════════

def train(csv_path="nifty_rl_ready.csv", total_timesteps=300_000, n_envs=4, seed=42,
          save_dir="./nifty_combined_model/", lot_size=DEFAULT_LOT_SIZE,
          initial_capital=100_000.0, eval_freq=5000):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    import torch

    print("\n" + "=" * 70)
    print("  NIFTY COMBINED OPTIONS RL TRAINER (CE + PE BUYING)")
    print("  FIX: train/test split happens BEFORE normalization is fitted")
    print("=" * 70)

    raw_train, raw_test, norm_train, norm_test, norm_params = DataLoader.load_and_prepare_split(
        csv_path, train_ratio=0.8
    )

    if len(raw_train) < 200:
        print(f"  Not enough training data ({len(raw_train)} bars).")
        return None, None

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("./checkpoints_combined/", exist_ok=True)
    os.makedirs("./logs_combined/", exist_ok=True)
    os.makedirs("./tb_combined/", exist_ok=True)

    def make_train_env():
        return Monitor(NiftyCombinedOptionsEnv(raw_train, norm_train, initial_capital, lot_size))

    def make_eval_env():
        return Monitor(NiftyCombinedOptionsEnv(raw_test, norm_test, initial_capital, lot_size))

    train_vec = make_vec_env(make_train_env, n_envs=n_envs, seed=seed)
    train_vec = VecNormalize(train_vec, norm_obs=False, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_vec = make_vec_env(make_eval_env, n_envs=1, seed=seed + 100)
    eval_vec = VecNormalize(eval_vec, norm_obs=False, norm_reward=False, training=False, clip_obs=10.0)

    sample_env = NiftyCombinedOptionsEnv(raw_train[:10], norm_train[:10], initial_capital, lot_size)
    obs_dim = sample_env.observation_space.shape[0]
    del sample_env

    model = PPO(
        policy="MlpPolicy", env=train_vec, seed=seed,n_steps=1024, batch_size=256,
        n_epochs=10, gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
        vf_coef=0.5, max_grad_norm=0.5, learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256, 128], activation_fn=torch.nn.Tanh),
        verbose=1, tensorboard_log="./tb_combined/",
    )

    callbacks = [
        TradingMetricsCallback(log_freq=2000),
        EvalCallback(eval_vec, best_model_save_path=save_dir, log_path="./logs_combined/",
                     eval_freq=max(eval_freq // n_envs, 1), n_eval_episodes=3, deterministic=True),
        CheckpointCallback(save_freq=max(25_000 // n_envs, 1), save_path="./checkpoints_combined/",
                            name_prefix="nifty_combined"),
    ]

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks),
                reset_num_timesteps=True, progress_bar=True)

    final_path = os.path.join(save_dir, "nifty_combined_final")
    model.save(final_path)
    train_vec.save(os.path.join(save_dir, "vecnormalize.pkl"))
    norm_params.save(os.path.join(save_dir, "norm_params.pkl"))

    feature_cols = NiftyCombinedOptionsEnv(raw_train[:10], norm_train[:10], lot_size=lot_size).feature_columns
    with open(os.path.join(save_dir, "feature_columns.json"), 'w') as f:
        json.dump(feature_cols, f)

    config = {
        'lot_size': lot_size, 'initial_capital': initial_capital,
        'train_date_range': [str(raw_train.index.min()), str(raw_train.index.max())],
        'test_date_range': [str(raw_test.index.min()), str(raw_test.index.max())],
    }
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print("\n  TRAINING COMPLETE.")
    print(f"  Best model:   {save_dir}/best_model.zip")
    print(f"  Norm params:  {save_dir}/norm_params.pkl  (fit on TRAIN split only)")

    return model, train_vec


def evaluate(csv_path="nifty_rl_ready.csv", model_path="./nifty_combined_model/best_model",
             save_dir="./nifty_combined_model/", lot_size=DEFAULT_LOT_SIZE, initial_capital=100_000.0):
    from stable_baselines3 import PPO

    norm_params_path = os.path.join(save_dir, "norm_params.pkl")
    if not os.path.exists(norm_params_path):
        print(f"  No saved norm_params found. Run training first.")
        return None
    norm_params = NormalizationParams.load(norm_params_path)

    raw_train, raw_test, _, _, _ = DataLoader.load_and_prepare_split(csv_path, train_ratio=0.8)
    norm_test = DataLoader._normalize_features(raw_test, norm_params, is_training=False)

    env = NiftyCombinedOptionsEnv(raw_test, norm_test, initial_capital, lot_size)
    model = PPO.load(model_path)

    backtester = Backtester(model, env, deterministic=True)
    results = backtester.run()
    backtester.print_report(results)
    return results


def test_env(csv_path="nifty_rl_ready.csv"):
    raw_train, raw_test, norm_train, norm_test, norm_params = DataLoader.load_and_prepare_split(
        csv_path, train_ratio=0.8
    )
    env = NiftyCombinedOptionsEnv(raw_train, norm_train, 100_000.0, DEFAULT_LOT_SIZE)

    obs, info = env.reset()
    print(f"  Obs shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]")

    steps, done = 0, False
    while not done and steps < 200:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated

    print(f"  Ran {steps} random steps. Final PnL: Rs.{info.get('realized_pnl', 0):,.2f}")
    print("  Environment test passed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test", choices=["test", "train", "evaluate", "backtest"])
    parser.add_argument("--csv", type=str, default="nifty_rl_ready.csv")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="./nifty_combined_model/")
    parser.add_argument("--lot-size", type=int, default=DEFAULT_LOT_SIZE)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--eval-freq", type=int, default=5000)
    args = parser.parse_args()

    if args.mode == "test":
        test_env(csv_path=args.csv)
    elif args.mode == "train":
        train(csv_path=args.csv, total_timesteps=args.timesteps, n_envs=args.envs, seed=args.seed,
              save_dir=args.model_dir, lot_size=args.lot_size, initial_capital=args.capital,
              eval_freq=args.eval_freq)
    elif args.mode in ("evaluate", "backtest"):
        evaluate(csv_path=args.csv, model_path=os.path.join(args.model_dir, "best_model"),
                 save_dir=args.model_dir, lot_size=args.lot_size, initial_capital=args.capital)