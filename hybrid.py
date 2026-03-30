from __future__ import annotations

import numpy as np
import pandas as pd


def _default_hybrid_params(meter_id: str) -> dict:
    if meter_id == 'm1':
        return {
            'normal_start': 6.0,
            'normal_end': 18.75,
            'stable_start': 8.25,
            'stable_end': 16.25,
            'pos_base': 80.0,
            'neg_base': -120.0,
            'off_median_cap': 25.0,
            'off_p90_offset': 25.0,
            'off_med_offset': 60.0,
            'off_neighbor_mult': 0.85,
            'off_extra_margin': 80.0,
            'drop_active_median_floor': 200.0,
            'drop_p10_offset': 20.0,
            'drop_med_mult': 0.25,
            'drop_neighbor_mult': 1.15,
            'spike_med_offset': 30.0,
            'scale_floor': 40.0,
            'score_offset': 0.35,
            'drop_score_offset': 0.30,
            'skip_negative_export': False,
        }
    return {
        'normal_start': 6.5,
        'normal_end': 20.0,
        'stable_start': 7.0,
        'stable_end': 18.5,
        'pos_base': 60.0,
        'neg_base': -70.0,
        'off_median_cap': 70.0,
        'off_p90_offset': 35.0,
        'off_med_offset': 50.0,
        'off_neighbor_mult': 0.85,
        'off_extra_margin': 80.0,
        'drop_active_median_floor': 110.0,
        'drop_p10_offset': 25.0,
        'drop_med_mult': 0.45,
        'drop_neighbor_mult': 1.15,
        'spike_med_offset': 40.0,
        'scale_floor': 40.0,
        'score_offset': 0.35,
        'drop_score_offset': 0.30,
        'skip_negative_export': True,
    }

def fit_hybrid_reference(train_df: pd.DataFrame, meter_id: str, params: dict | None = None) -> dict:
    cfg = _default_hybrid_params(meter_id)
    if params:
        cfg.update(params)

    d_p = train_df.loc[train_df['valid_for_scoring'].fillna(False), 'P_import'].diff()
    pos_diff_q = float(d_p.quantile(0.997)) if d_p.notna().any() else float(cfg['pos_base'])
    neg_diff_q = float(d_p.quantile(0.003)) if d_p.notna().any() else float(cfg['neg_base'])

    return {
        'pos_diff_q': pos_diff_q,
        'neg_diff_q': neg_diff_q,
    }

def build_hybrid_score_frame(
    df: pd.DataFrame,
    meter_id: str,
    params: dict | None = None,
    reference: dict | None = None,
) -> pd.DataFrame:
    cfg = _default_hybrid_params(meter_id)
    if params:
        cfg.update(params)

    out = df.copy().sort_values('Timestamp').reset_index(drop=True)
    out['current_score'] = 0.0
    out['hybrid_score'] = 0.0
    out['dP'] = out['P_import'].diff()

    normal_window = out['hour'].between(cfg['normal_start'], cfg['normal_end'])
    stable_window = out['hour'].between(cfg['stable_start'], cfg['stable_end'])
    if reference is None:
        reference = fit_hybrid_reference(out, meter_id, params)

    pos_diff_q = float(reference.get('pos_diff_q', cfg['pos_base']))
    neg_diff_q = float(reference.get('neg_diff_q', cfg['neg_base']))

    for i, row in out.iterrows():
        if not bool(row.get('valid_for_scoring', False)) or pd.isna(row.get('median')):
            continue
        if cfg.get('skip_negative_export', False) and row.get('is_negative_export', 0) == 1:
            continue

        p = float(row['P_import'])
        med = float(row['median'])
        p90 = float(row['p90']) if pd.notna(row.get('p90')) else med
        p10 = float(row['p10']) if pd.notna(row.get('p10')) else med * 0.8
        scale = max(float(row['scale']) if pd.notna(row.get('scale')) else 0.0, float(cfg['scale_floor']))
        d_p = float(row['dP']) if pd.notna(row.get('dP')) else 0.0
        score = 0.0

        if (not bool(normal_window.iloc[i])) and med < float(cfg['off_median_cap']):
            threshold = max(p90 + float(cfg['off_p90_offset']), med + float(cfg['off_med_offset']))
            if p > threshold:
                prev = i > 0 and out.at[i - 1, 'P_import'] > threshold * float(cfg['off_neighbor_mult'])
                nxt = i + 1 < len(out) and out.at[i + 1, 'P_import'] > threshold * float(cfg['off_neighbor_mult'])
                if prev or nxt or p > threshold + float(cfg['off_extra_margin']):
                    score = max(score, min(1.0, (p - threshold) / scale + float(cfg['score_offset'])))

        if bool(stable_window.iloc[i]) and med > float(cfg['drop_active_median_floor']):
            low_thr = min(p10 - float(cfg['drop_p10_offset']), med * float(cfg['drop_med_mult']))
            low_thr = max(low_thr, 0.0)
            if p < low_thr:
                prev = i > 0 and out.at[i - 1, 'P_import'] < low_thr * float(cfg['drop_neighbor_mult'])
                nxt = i + 1 < len(out) and out.at[i + 1, 'P_import'] < low_thr * float(cfg['drop_neighbor_mult'])
                if prev or nxt:
                    score = max(score, min(1.0, (low_thr - p) / scale + float(cfg['drop_score_offset'])))

        spike_thr = max(pos_diff_q, float(cfg['pos_base']))
        if bool(stable_window.iloc[i]) and d_p > spike_thr and p > med + float(cfg['spike_med_offset']):
            score = max(score, min(1.0, (d_p - spike_thr) / scale + float(cfg['score_offset'])))

        drop_dp_thr = min(neg_diff_q, float(cfg['neg_base']))
        if bool(stable_window.iloc[i]) and d_p < drop_dp_thr and p < p10:
            score = max(score, min(1.0, (abs(d_p) - abs(drop_dp_thr)) / scale + float(cfg['score_offset'])))

        out.at[i, 'hybrid_score'] = score
        out.at[i, 'current_score'] = score

    return out