import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import os

print("BOOTING PURE RENDERER...")
run_dir = "outputs/honest_matern"

# 1. Load the data
try:
    df = pd.read_csv(f"{run_dir}/MONOLITH_DATA.csv")
    features = np.load(f"{run_dir}/features.npy")
    meta = pd.read_csv(f"{run_dir}/article_metadata.csv")
except Exception as e:
    print(f"CRITICAL ERROR LOADING DATA: {e}")
    exit()

# 2. Force a mathematically pure 3D projection
print("Calculating 3D Riemannian Geometry (PCA)...")
pca = PCA(n_components=3)
proj = pca.fit_transform(features)
df['X'] = proj[:, 0]
df['Y'] = proj[:, 1]
df['Z'] = proj[:, 2]

# 3. Fuse the spine (Metadata to Physics via bt_uid)
print("Fusing ID Lineage and Bias Stamps...")
if 'bt_uid' in df.columns and 'bt_uid' in meta.columns:
    df = df.merge(meta, on='bt_uid', how='left')
else:
    print("WARNING: bt_uid missing. Merging blindly on index.")
    df = df.join(meta, rsuffix='_drop')

# 4. Format the Hover Text
bias_col = 'affiliation' if 'affiliation' in df.columns else 'bias' if 'bias' in df.columns else None
df['Bias'] = df[bias_col] if bias_col else "Unknown"
text_col = 'title' if 'title' in df.columns else 'text' if 'text' in df.columns else None
df['Snippet'] = df[text_col].astype(str).str[:80] + "..." if text_col else "No text"

# 5. Draw the Truth
print("Rendering UI...")
fig = px.scatter_3d(
    df, x='X', y='Y', z='Z',
    color='verdict',
    color_discrete_map={'HONEST': '#00F0FF', 'TAUTOLOGY': '#FFFFCC', 'PHANTOM': '#FF0044'},
    hover_name='Bias',
    hover_data={'X': False, 'Y': False, 'Z': False, 'verdict': True, 'Snippet': True, 'bt_uid': True},
    title="SYSTEM 2: THE GEOMETRIC LAYER (Absolute Cost)",
    template="plotly_dark"
)

fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
fig.show()
print("RENDER COMPLETE. Check your browser.")
