import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

def load_and_merge_mp_data():
    import re
    
    # 1. Load Lok Sabha
    ls_df = pd.read_csv('data/lok_sabha_allocation.csv', encoding='utf-8-sig')
    ls_df.columns = ['sr_no', 'state', 'mp_name', 'constituency', 'fund_sanctioned']
    ls_df['house'] = "Lok Sabha"
    ls_df['elected_or_nominated'] = np.nan
    
    # 2. Load Rajya Sabha
    rs_df = pd.read_csv('data/rajya_sabha_allocation.csv', encoding='utf-8-sig')
    rs_df.columns = ['sr_no', 'state', 'mp_name', 'elected_or_nominated', 'fund_sanctioned']
    rs_df['mp_name'] = rs_df['mp_name'].apply(lambda name: re.sub(r'\s*\(.*?\)', '', str(name)).strip())
    rs_df['constituency'] = rs_df['state']
    rs_df['house'] = "Rajya Sabha"
    
    # 3. Same columns order before merging
    columns_order = ['sr_no', 'state', 'mp_name', 'constituency', 'fund_sanctioned', 'house', 'elected_or_nominated']
    ls_df = ls_df[columns_order]
    rs_df = rs_df[columns_order]
    
    # 4. Concatenate and reset sr_no
    combined = pd.concat([ls_df, rs_df], ignore_index=True)
    combined['sr_no'] = range(1, len(combined) + 1)
    
    # 5. Convert to numeric and drop NaNs
    combined['fund_sanctioned'] = pd.to_numeric(combined['fund_sanctioned'], errors='coerce')
    combined = combined.dropna(subset=['fund_sanctioned']).reset_index(drop=True)
    
    return combined

def get_data():
    try:
        df = load_and_merge_mp_data()
    except FileNotFoundError:
        # Fallback for testing if file doesn't exist
        print("Warning: data/lok_sabha_allocation.csv or data/rajya_sabha_allocation.csv not found. Using generated mock data.")
        df = pd.DataFrame({
            'sr_no': range(1, 101),
            'state': ['State A', 'State B'] * 50,
            'mp_name': ['MP ' + str(i%10) for i in range(100)],
            'constituency': ['Const ' + str(i) for i in range(100)],
            'fund_sanctioned': np.random.uniform(5, 25, 100), # in Cr
            'house': ['Lok Sabha', 'Rajya Sabha'] * 50,
            'elected_or_nominated': [np.nan, 'Elected'] * 50
        })
        
    np.random.seed(42)
    n_rows = len(df)
    
    # Generate synthetic execution data
    df['fund_utilized'] = df['fund_sanctioned'] * np.random.uniform(0.40, 0.99, n_rows)
    df['physical_progress_percent'] = np.random.uniform(5, 95, n_rows)
    df['months_since_sanction'] = np.random.randint(3, 37, n_rows)
    
    # --- Add after generating physical_progress_percent ---
    df['days_pending_sanction'] = np.random.randint(5, 120, n_rows) # Administrative latency
    df['unspent_balance'] = df['fund_sanctioned'] - df['fund_utilized'] # Idle funds
    
    vendors = [
        "Apex Builders", "Global Infra", "Nirman Tech", "Pioneer Constructions", 
        "Vertex Engineering", "Quantum Works", "Synergy Projects", "Stellar Infra", 
        "Titan Builders", "Omega Developers", "Crest Constructions", "Zenith Infra", 
        "Nova Projects", "Prime Builders", "Nexus Engineering"
    ]
    df['vendor_name'] = np.random.choice(vendors, n_rows)
    
    # Introduce anomalies (~15% of rows)
    anomaly_indices = np.random.choice(df.index, size=int(n_rows * 0.15), replace=False)
    # High utilization + low progress
    df.loc[anomaly_indices, 'fund_utilized'] = df.loc[anomaly_indices, 'fund_sanctioned'] * np.random.uniform(0.91, 0.99, len(anomaly_indices))
    df.loc[anomaly_indices, 'physical_progress_percent'] = np.random.uniform(5, 15, len(anomaly_indices))
    
    # Vendor repetition count (per MP)
    df['vendor_repetition_count'] = df.groupby(['mp_name', 'vendor_name'])['vendor_name'].transform('count')
    
    # Feature engineering
    df['fund_utilization_ratio'] = df['fund_utilized'] / df['fund_sanctioned']
    df['progress_gap'] = df['fund_utilization_ratio'] - (df['physical_progress_percent'] / 100)
    
    # Anomaly detection using Isolation Forest
    features = ['fund_utilization_ratio', 'progress_gap', 'months_since_sanction', 'vendor_repetition_count']
    X = df[features]
    
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    preds = iso_forest.fit_predict(X)
    df['is_anomaly'] = preds == -1
    # anomaly_score -> lower score means more anomalous in sklearn, but you can use it directly
    df['anomaly_score'] = iso_forest.decision_function(X)
    
    # Rule flags
    def get_rules(row):
        rules = []
        if row['fund_utilization_ratio'] > 0.9 and (row['physical_progress_percent'] / 100) < 0.2:
            rules.append("Fund Misuse")
        if row['vendor_repetition_count'] > 4:
            rules.append("Vendor Concentration")
        if row['months_since_sanction'] > 12 and (row['physical_progress_percent'] / 100) < 0.3:
            rules.append("Execution Delay")
        if row['days_pending_sanction'] > 60:
            rules.append(f"Admin Delay ({int(row['days_pending_sanction'])} days)")
        if (row['unspent_balance'] / row['fund_sanctioned']) > 0.7 and row['months_since_sanction'] > 18:
            rules.append("Idle Fund Hoarding")
        return rules

    df['rules_triggered'] = df.apply(get_rules, axis=1)
    df['rule_count'] = df['rules_triggered'].apply(len)
    df['rules_str'] = df['rules_triggered'].apply(lambda x: ", ".join(x) if x else "None")
    
    # Risk Level
    def determine_risk(row):
        if row['is_anomaly'] and row['rule_count'] >= 2:
            return "High"
        elif row['is_anomaly'] or row['rule_count'] == 1:
            return "Medium"
        else:
            return "Low"
            
    df['risk_level'] = df.apply(determine_risk, axis=1)
    
    return df
