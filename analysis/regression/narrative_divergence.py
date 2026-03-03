"""
analysis/narrative_divergence.py - The Thesis Metric for Titan Protocol.

Measures "Objective Fracture" by calculating the Euclidean distance between
Blue and Red team centroids on specific topics, normalized by global standard deviation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

# --- Ground Truth Mapping (reusing from ablation.py) ---
def get_tribe(publication: str) -> int:
    """
    Maps publication names to tribe labels (0: RED, 1: BLUE, 2: GRAY).
    """
    if not isinstance(publication, str):
        return 2 # GRAY TEAM for non-string (e.g., NaN) inputs
        
    publication = publication.lower()
    
    red_team = [
        "al jazeera", "electronic intifada", "haaretz", "guardian", 
        "intercept", "mondoweiss", "middle east eye", "mintpress news"
    ]
    blue_team = [
        "arutz sheva", "jpost", "israel hayom", "fox news", "kohelet policy forum",
        "breitbart", "daily wire", "townhall"
    ]
    
    if any(p in publication for p in red_team):
        return 0  # RED TEAM
    elif any(p in publication for p in blue_team):
        return 1  # BLUE TEAM
    else:
        return 2  # GRAY TEAM (e.g., Reuters, BBC, CNN, NYT)

# --- Topic Extraction (simplified for synthetic data) ---
def get_topic(title: str) -> str:
    """
    Infers topic from article title (simplified for synthetic data).
    For real data, this would use a more robust topic modeling approach.
    """
    title = title.lower()
    if "judicial reform" in title or "supreme court" in title or "basic law" in title:
        return "Judicial Reform"
    elif "jenin" in title or "idf raid" in title:
        return "Jenin Raid"
    elif "huvara" in title or "settler violence" in title or "riots" in title:
        return "Huvara Riots"
    elif "gaza" in title or "rocket fire" in title or "border escalation" in title:
        return "Gaza Escalation"
    elif "settlement expansion" in title or "area c" in title:
        return "Settlement Expansion"
    elif "al-aqsa" in title or "temple mount" in title:
        return "Al-Aqsa/Temple Mount"
    elif "saudi-israel" in title or "normalization" in title:
        return "Saudi-Israel Normalization"
    elif "civil rights" in title or "apartheid" in title or "international law" in title:
        return "Civil Rights/Apartheid"
    return "General Conflict" # Default topic if none match

def calculate_narrative_divergence(data_path: Path, output_dir: Path) -> pd.DataFrame:
    """
    Calculates the narrative divergence (fracture) between Blue and Red teams per topic.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Clean data: drop rows with NaN in critical columns
    df_clean = df.dropna(subset=['x_proj', 'y_proj', 'z_height', 'source', 'title'])
    
    if df_clean.empty:
        print("Warning: Cleaned DataFrame is empty after dropping NaNs. Cannot perform analysis.")
        return pd.DataFrame(columns=['Topic', 'Fracture_Distance', 'Consensus_Score', 'Normalized_Fracture_ZScore'])

    # Add tribe_label
    df_clean['tribe_label'] = df_clean['source'].apply(get_tribe)
    
    # Add topic
    df_clean['topic'] = df_clean['title'].apply(get_topic)

    # Filter for Red (0) and Blue (1) teams only for fracture analysis
    df_red_blue = df_clean[df_clean['tribe_label'].isin([0, 1])].copy()
    
    if df_red_blue.empty:
        print("Warning: No Red or Blue team articles found after filtering. Cannot calculate fracture.")
        return pd.DataFrame(columns=['Topic', 'Fracture_Distance', 'Consensus_Score', 'Normalized_Fracture_ZScore'])

    # Coordinates for centroid calculation
    coords_cols = ['x_proj', 'y_proj', 'z_height']

    fracture_data = []

    # Group by topic and calculate centroids
    for topic_name, group_df in df_red_blue.groupby('topic'):
        red_articles = group_df[group_df['tribe_label'] == 0]
        blue_articles = group_df[group_df['tribe_label'] == 1]

        if len(red_articles) > 0 and len(blue_articles) > 0:
            centroid_red = red_articles[coords_cols].mean().values
            centroid_blue = blue_articles[coords_cols].mean().values

            # Calculate Euclidean Distance
            fracture_distance = np.linalg.norm(centroid_blue - centroid_red)
            fracture_data.append({'Topic': topic_name, 'Fracture_Distance': fracture_distance})
        elif len(red_articles) == 0 and len(blue_articles) == 0:
            # Both empty, no data for this topic
            continue
        else:
            # One team is missing, can't calculate a meaningful fracture
            # Assign NaN or some indicator for topics where only one team is present
            fracture_data.append({'Topic': topic_name, 'Fracture_Distance': np.nan})

    fracture_df = pd.DataFrame(fracture_data)
    
    if fracture_df.empty:
        print("Warning: No topics with both Red and Blue team articles found. Cannot calculate fracture.")
        return pd.DataFrame(columns=['Topic', 'Fracture_Distance', 'Consensus_Score', 'Normalized_Fracture_ZScore'])

    # Drop NaNs introduced by topics with only one team
    fracture_df.dropna(subset=['Fracture_Distance'], inplace=True)
    
    if fracture_df.empty:
        print("Warning: No valid fracture distances calculated after dropping NaNs. Cannot calculate fracture.")
        return pd.DataFrame(columns=['Topic', 'Fracture_Distance', 'Consensus_Score', 'Normalized_Fracture_ZScore'])


    # Normalize Fracture_Distance by global standard deviation (Z-score)
    global_std = df_red_blue[coords_cols].std().mean() # Mean std of all coordinates
    if global_std > 0:
        scaler = StandardScaler()
        fracture_df['Normalized_Fracture_ZScore'] = scaler.fit_transform(
            fracture_df[['Fracture_Distance']]
        )
    else:
        fracture_df['Normalized_Fracture_ZScore'] = 0.0 # No variance, so no fracture

    # Consensus Score (Inverse of Fracture)
    fracture_df['Consensus_Score'] = 1.0 / (fracture_df['Fracture_Distance'] + 1e-8) # Add epsilon

    fracture_df.sort_values(by='Fracture_Distance', ascending=False, inplace=True)

    # Output to CSV
    output_path = output_dir / "fracture_scores.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    fracture_df.to_csv(output_path, index=False)
    print(f"\nFracture scores saved to: {output_path}")

    # Print summary
    print("\n--- Narrative Divergence Analysis ---")
    if not fracture_df.empty:
        most_fractured = fracture_df.iloc[0]
        most_agreed = fracture_df.iloc[-1]
        print(f"Most Fractured Event: '{most_fractured['Topic']}' (Fracture: {most_fractured['Fracture_Distance']:.2f}, Z-Score: {most_fractured['Normalized_Fracture_ZScore']:.2f})")
        print(f"Most Agreed Event: '{most_agreed['Topic']}' (Fracture: {most_agreed['Fracture_Distance']:.2f}, Z-Score: {most_agreed['Normalized_Fracture_ZScore']:.2f})")
    else:
        print("No valid narrative divergence data to report.")

    return fracture_df

if __name__ == "__main__":
    # Assuming MONOLITH_DATA.csv is in experiments_20260213_072109/synthetic/rbf_seed43/
    root_path = Path(__file__).resolve().parent.parent # V3
    data_path = root_path / "experiments_20260213_072109" / "synthetic" / "rbf_seed43" / "MONOLITH_DATA.csv"
    output_dir = root_path / "analysis" / "outputs"
    
    fracture_results = calculate_narrative_divergence(data_path, output_dir)
    print("
Full Fracture Results:")
    print(fracture_results)