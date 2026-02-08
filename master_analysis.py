import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# CONFIGURATION

# FILTERS
ATTENTION_CHECK_TARGET = 5      # Value for "Somewhat Agree" on Q2_8
MIN_DURATION_SECONDS = 90       # Minimum time


SHOPPING_COL = 'Q5'
MIN_SHOPPING_FREQ = 6

# MANIPULATION CHECK SETTINGS 
AUDIO_TARGET_VAL = 3  
SILENT_TARGET_VAL = 5  

# Columns
flow_items = ['Q2_1', 'Q2_2', 'Q2_3', 'Q2_4', 'Q2_5', 'Q2_6', 'Q2_7', 'Q2_9', 'Q2_10']
attn_col = 'Q2_8'
manip_col = 'Q1' # "Did you notice a sound?"

# 2. CLEANING FUNCTION
def clean_and_process(filename, label, start_date_cutoff, target_manipulation_val):
    df = pd.read_csv(filename, skiprows=[1, 2])
    df['RecordedDate'] = pd.to_datetime(df['RecordedDate'])
    
    #DATE FILTER this is for the people we did before we fixed the survey after your feedback.
    n_original = len(df)
    df_clean = df[df['RecordedDate'] >= start_date_cutoff].copy()
    dropped_pilot = n_original - len(df_clean)
    
    #Coerce numeric (Added SHOPPING_COL here)
    cols_to_numeric = flow_items + [attn_col, manip_col, 'Duration (in seconds)', 'Finished', SHOPPING_COL]
    for col in cols_to_numeric:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    #FINISHED & DURATION
    df_clean = df_clean[df_clean['Finished'] == 1]
    n_before_speed = len(df_clean)
    df_clean = df_clean[df_clean['Duration (in seconds)'] >= MIN_DURATION_SECONDS]
    dropped_speed = n_before_speed - len(df_clean)

    #ATTENTION CHECK
    n_before_attn = len(df_clean)
    df_clean = df_clean[df_clean[attn_col] == ATTENTION_CHECK_TARGET]
    dropped_attn = n_before_attn - len(df_clean)

    #MANIPULATION CHECK (Did they hear the sound?)
    n_before_manip = len(df_clean)
    df_clean = df_clean[df_clean[manip_col] == target_manipulation_val]
    dropped_manip = n_before_manip - len(df_clean)

    #SHOPPING FREQUENCY FILTER
    n_before_freq = len(df_clean)
    df_clean = df_clean[df_clean[SHOPPING_COL] <= MIN_SHOPPING_FREQ]
    dropped_freq = n_before_freq - len(df_clean)

    # Calculate Flow
    df_clean['Flow_Score'] = df_clean[flow_items].mean(axis=1)
    df_clean['Condition'] = label
    
    print(f"--- {label} Cleaning Report ---")
    print(f"Original N: {n_original}")
    print(f"Date Filter Removed: {dropped_pilot}")
    print(f"Speedsters Removed: {dropped_speed}")
    print(f"Infrequent Shoppers Removed: {dropped_freq}")
    print(f"Attention Check Failed: {dropped_attn}")
    print(f"Manipulation Check Failed: {dropped_manip}")
    print(f"FINAL VALID N: {len(df_clean)}\n")
    
    return df_clean

# 3. EXECUTION
file_soundless = 'Qualtrics_Survey_Soundless.csv' 
file_sound = 'Qualtrics_Survey_Sound.csv'

try:
    # SILENT GROUP
    df_silent = clean_and_process(
        file_soundless, 
        'Silent', 
        start_date_cutoff='2026-01-01',
        target_manipulation_val=SILENT_TARGET_VAL 
    )
    
    # AUDIO GROUP
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
        plt.title("Flow Scores (Participants who PASSED Checks)")
        plt.show()

except Exception as e:
    print(f"Error: {e}")

if 'df_all' in locals() and len(df_all) > 10: 
    print("\n" + "="*30)
    print("--- EXPLORATORY MODERATOR ANALYSIS ---")
    print("="*30)

    # 1. Prepare Variables
    df_all['Condition_Code'] = df_all['Condition'].apply(lambda x: 1 if x == 'Auditory' else 0)
    
    df_all = df_all.rename(columns={
        'Q4': 'Gender', 
        'Q5': 'Shopping_Freq',
        'Q3': 'Age_Num'
    })

    # --- MODEL A: Gender ---
    try:
        model_gender = smf.ols("Flow_Score ~ Condition_Code * C(Gender)", data=df_all).fit()
        print("\n[ Interaction Check: GENDER ]")
        print(model_gender.summary().tables[1])
    except Exception as e:
        print(f"Gender analysis failed: {e}")

    # --- MODEL B: Shopping Frequency ---
    try:
        model_freq = smf.ols("Flow_Score ~ Condition_Code * Shopping_Freq", data=df_all).fit()
        print("\n[ Interaction Check: SHOPPING FREQUENCY ]")
        print(model_freq.summary().tables[1])
    except Exception as e:
        print(f"Freq analysis failed: {e}")

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    sns.pointplot(data=df_all, x='Condition', y='Flow_Score', hue='Gender', errorbar='se')
    plt.title("Interaction: Gender")

    plt.subplot(1, 2, 2)
    sns.regplot(data=df_all[df_all['Condition']=='Silent'], x='Shopping_Freq', y='Flow_Score', color='blue', label='Silent', scatter_kws={'alpha':0.3})
    sns.regplot(data=df_all[df_all['Condition']=='Auditory'], x='Shopping_Freq', y='Flow_Score', color='orange', label='Audio', scatter_kws={'alpha':0.3})
    plt.legend()
    plt.title("Interaction: Shopping Freq")
    
    plt.tight_layout()
    plt.show()