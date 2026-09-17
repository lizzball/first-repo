from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_terminal_fleet(
    seed: int = 42,
    n_terminals: int = 4_000,
    days: int = 30,
) -> pd.DataFrame:
    """Generate a synthetic fleet with a small number of deteriorating terminals.

    The simulation is intentionally generic. It is designed to create a realistic
    rare-event classification problem, not to reproduce Square production telemetry.
    """
    rng = np.random.default_rng(seed)
    terminal = np.arange(n_terminals)

    firmware = rng.choice(
        ["stable_1", "stable_2", "watchlist"],
        size=n_terminals,
        p=[0.45, 0.40, 0.15],
    )
    seller_type = rng.choice(
        ["retail", "food_service", "mobile_market", "services"],
        size=n_terminals,
        p=[0.35, 0.25, 0.20, 0.20],
    )

    base_wifi = rng.normal(-58, 5.5, n_terminals)
    base_wifi -= np.where(seller_type == "mobile_market", 3.5, 0)
    base_volume = np.exp(rng.normal(np.log(80), 0.65, n_terminals))
    battery_health = np.clip(rng.normal(0.88, 0.07, n_terminals), 0.60, 1.0)

    event_probability = np.where(firmware == "watchlist", 0.78, 0.48)
    has_event = rng.random(n_terminals) < event_probability
    event_start = np.where(has_event, rng.integers(7, 23, size=n_terminals), 99)
    failure_day = event_start + 6
    event_strength = np.where(has_event, rng.uniform(0.8, 1.35, n_terminals), 0)

    frames: list[pd.DataFrame] = []

    for day in range(days):
        weekend = int(day % 7 in [5, 6])

        degradation = np.clip((day - event_start + 1) / 6, 0, 1) * event_strength
        degradation = np.where(day >= failure_day, 0, degradation)

        market_boost = np.where(
            (seller_type == "mobile_market") & bool(weekend),
            1.8,
            1.0,
        )
        transactions = np.maximum(
            1,
            rng.poisson(np.clip(base_volume * market_boost, 1, 1_000)),
        )

        wifi_rssi = base_wifi - 8.5 * degradation + rng.normal(0, 2.2, n_terminals)
        weak_network = np.clip((-61 - wifi_rssi) / 14, 0, 2.5)

        reconnects = rng.poisson(
            np.clip(0.2 + 1.3 * weak_network + 3.0 * degradation, 0.05, 15)
        )
        latency_s = np.clip(
            rng.normal(1.15, 0.15, n_terminals)
            + 0.32 * weak_network
            + 0.9 * degradation
            + 0.0012 * transactions,
            0.3,
            8,
        )
        timeout_rate = np.clip(
            rng.beta(1.15, 85, size=n_terminals)
            + 0.010 * reconnects
            + 0.060 * degradation
            + 0.010 * weak_network,
            0,
            0.6,
        )
        battery_risk = np.clip(
            1 - battery_health + 0.09 * degradation + rng.normal(0, 0.02, n_terminals),
            0,
            1,
        )
        charge_interrupts = rng.poisson(
            0.05 + 0.45 * degradation + 0.08 * (battery_health < 0.75)
        )

        planned_failure = has_event & (day == failure_day - 1)
        spontaneous_failure = (~planned_failure) & (rng.random(n_terminals) < 0.0045)
        failure_next_24h = (planned_failure | spontaneous_failure).astype(int)

        support_contact = rng.binomial(
            1,
            np.clip(0.12 + 0.58 * failure_next_24h + 0.65 * timeout_rate, 0, 1),
        )

        frames.append(
            pd.DataFrame(
                {
                    "terminal_id": [f"T{i:05d}" for i in terminal],
                    "day": day,
                    "firmware": firmware,
                    "seller_type": seller_type,
                    "wifi_rssi": wifi_rssi,
                    "reconnects": reconnects,
                    "latency_s": latency_s,
                    "timeout_rate": timeout_rate,
                    "transactions": transactions,
                    "battery_risk": battery_risk,
                    "charge_interrupts": charge_interrupts,
                    "weekend": weekend,
                    "failure_next_24h": failure_next_24h,
                    "support_contact": support_contact,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trailing 7, 14 and 30 day telemetry summaries without future leakage."""
    out = df.sort_values(["terminal_id", "day"]).copy()
    grouped = out.groupby("terminal_id", group_keys=False)

    history_columns = ["wifi_rssi", "reconnects", "latency_s", "timeout_rate"]
    for window in [7, 14, 30]:
        for column in history_columns:
            out[f"{column}_{window}d_mean"] = grouped[column].transform(
                lambda s: s.shift(1).rolling(window, min_periods=3).mean()
            )

        out[f"wifi_delta_{window}d"] = (
            out["wifi_rssi"] - out[f"wifi_rssi_{window}d_mean"]
        )
        out[f"timeout_delta_{window}d"] = (
            out["timeout_rate"] - out[f"timeout_rate_{window}d_mean"]
        )

    return out
