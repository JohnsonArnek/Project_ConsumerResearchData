import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
# FILTERS
ATTENTION_CHECK_TARGET = 5      # Value for "Somewhat Agree" on Q2_8
MIN_DURATION_SECONDS = 90       # Minimum time (lowered slightly to be safe)

# MANIPULATION CHECK SETTINGS (CRITICAL: CHECK YOUR QUALTRICS VALUES)
# What number means "Yes, I heard it"? (For Audio Group)
AUDIO_TARGET_VAL = 3  
# What number means "No, I didn't hear it"? (For Silent Group)
# NOTE: Your data showed '5' for silent. Verify if 5 means "No" or "Strongly Disagree".
SILENT_TARGET_VAL = 5  

# Columns
flow_items = ['Q2_1', 'Q2_2', 'Q2_3', 'Q2_4', 'Q2_5', 'Q2_6', 'Q2_7', 'Q2_9', 'Q2_10']
attn_col = 'Q2_8'
manip_col = 'Q1' # "Did you notice a sound?"

# ---------------------------------------------------------
# 2. CLEANING FUNCTION
# ---------------------------------------------------------
def clean_and_process(filename, label, start_date_cutoff, target_manipulation_val):
    df = pd.read_csv(filename, skiprows=[1, 2])
    df['RecordedDate'] = pd.to_datetime(df['RecordedDate'])
    
    # 1. DATE FILTER
    n_original = len(df)
    df_clean = df[df['RecordedDate'] >= start_date_cutoff].copy()
    dropped_pilot = n_original - len(df_clean)
    
    # Coerce numeric
    cols_to_numeric = flow_items + [attn_col, manip_col, 'Duration (in seconds)', 'Finished']
    for col in cols_to_numeric:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 2. FINISHED & DURATION
    df_clean = df_clean[df_clean['Finished'] == 1]
    n_before_speed = len(df_clean)
    df_clean = df_clean[df_clean['Duration (in seconds)'] >= MIN_DURATION_SECONDS]
    dropped_speed = n_before_speed - len(df_clean)

    # 3. ATTENTION CHECK
    n_before_attn = len(df_clean)
    df_clean = df_clean[df_clean[attn_col] == ATTENTION_CHECK_TARGET]
    dropped_attn = n_before_attn - len(df_clean)

    # 4. MANIPULATION CHECK (Did they hear the sound?)
    # We remove people who gave the WRONG answer for their condition
    n_before_manip = len(df_clean)
    df_clean = df_clean[df_clean[manip_col] == target_manipulation_val]
    dropped_manip = n_before_manip - len(df_clean)

    # Calculate Flow
    df_clean['Flow_Score'] = df_clean[flow_items].mean(axis=1)
    df_clean['Condition'] = label
    
    print(f"--- {label} Cleaning Report ---")
    print(f"Original N: {n_original}")
    print(f"Date Filter Removed: {dropped_pilot}")
    print(f"Speedsters Removed: {dropped_speed}")
    print(f"Attention Check Failed: {dropped_attn}")
    print(f"Manipulation Check Failed: {dropped_manip} (Wrong answer to 'Did you notice sound?')")
    print(f"FINAL VALID N: {len(df_clean)}\n")
    
    return df_clean

# ---------------------------------------------------------
# 3. EXECUTION
# ---------------------------------------------------------
file_soundless = 'Qualtrics_Survey_Soundless.csv' 
file_sound = 'Qualtrics_Survey_Sound.csv'

try:
    # SILENT GROUP: 
    # Keep Jan data + Require them to answer SILENT_TARGET_VAL (e.g., No)
    df_silent = clean_and_process(
        file_soundless, 
        'Silent', 
        start_date_cutoff='2026-01-01',
        target_manipulation_val=SILENT_TARGET_VAL 
    )
    
    # AUDIO GROUP:
    # Remove Jan data + Require them to answer AUDIO_TARGET_VAL (e.g., Yes)
    df_audio = clean_and_process(
        file_sound, 
        'Auditory', 
        start_date_cutoff='2026-02-01',
        target_manipulation_val=AUDIO_TARGET_VAL
    )
    
    # Combine and Test
    if len(df_silent) < 2 or len(df_audio) < 2:
        print("ERROR: Not enough data left after filtering to run statistics!")
    else:
        df_all = pd.concat([df_silent, df_audio])
        
        # H1: Mann-Whitney U
        u_stat, p_h1 = stats.mannwhitneyu(df_audio['Flow_Score'], df_silent['Flow_Score'], alternative='greater')
        
        # H2: Levene's Test
        rank_audio = stats.rankdata(df_audio['Flow_Score'])
        rank_silent = stats.rankdata(df_silent['Flow_Score'])
        lev_stat, p_h2 = stats.levene(rank_audio, rank_silent, center='mean')

        print("--- FINAL RESULTS (Strict Filtering) ---")
        print(f"Silent Mean: {df_silent['Flow_Score'].mean():.2f} (SD: {df_silent['Flow_Score'].std():.2f})")
        print(f"Audio Mean:  {df_audio['Flow_Score'].mean():.2f} (SD: {df_audio['Flow_Score'].std():.2f})")
        print("-" * 30)
        print(f"H1 (Intensity) P-value: {p_h1:.4f}")
        print(f"H2 (Variance)  P-value: {p_h2:.4f}")
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Condition', y='Flow_Score', data=df_all, inner='stick', palette='muted')
        plt.title("Flow Scores (Participants who PASSED Manipulation Check)")
        plt.show()

except Exception as e:
    print(f"Error: {e}")