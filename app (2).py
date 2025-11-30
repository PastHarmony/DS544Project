import streamlit as st
pip install matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib

# --- Load the trained model and data ---
# Make sure these files are in the same directory as your app.py or provide full paths
try:
    gaia_mag_model = joblib.load('gaia_mag_random_forest_model.pkl')
    mag = pd.read_csv('mag_data_for_streamlit.csv')
except FileNotFoundError:
    st.error("Model or data file not found. Please ensure 'gaia_mag_random_forest_model.pkl' and 'mag_data_for_streamlit.csv' are in the same directory as this script.")
    st.stop()

# Prepare statistics for telescope suggestions from the 'mag' DataFrame
telescope_gaiamag_stats = mag[mag['sy_gaiamag'].notna()].groupby('disc_facility')['sy_gaiamag'].agg(
    mean_gaiamag='mean',
    std_gaiamag='std',
    count_gaiamag='count'
).reset_index()
telescope_gaiamag_stats['std_gaiamag'] = telescope_gaiamag_stats['std_gaiamag'].fillna(0)

# --- Functions for estimation and suggestion ---
def estimate_gaia_magnitude(planet_r, planet_m, star_r, star_m):
    if planet_m == 0 or star_m == 0:
        return "Error: Planet or stellar mass cannot be zero for ratio calculation."

    input_planet_mass_to_radius_ratio = planet_r / planet_m
    input_stellar_mass_to_radius_ratio = star_r / star_m

    X_predict = pd.DataFrame([[input_stellar_mass_to_radius_ratio, input_planet_mass_to_radius_ratio]],
                             columns=['stellar_mass_to_radius_ratio', 'planet_mass_to_radius_ratio'])

    try:
        estimated_gaiamag = gaia_mag_model.predict(X_predict)[0]
        return estimated_gaiamag
    except Exception as e:
        return f"Prediction error: {e}"

def suggest_telescopes(estimated_mag, num_suggestions=3):
    if not isinstance(estimated_mag, (int, float)):
        return []

    telescope_gaiamag_stats['diff_from_estimate'] = np.abs(telescope_gaiamag_stats['mean_gaiamag'] - estimated_mag)
    ranked_telescopes = telescope_gaiamag_stats.sort_values(by='diff_from_estimate').head(num_suggestions)

    return ranked_telescopes['disc_facility'].tolist()

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Exoplanet Transit Estimator")

st.title('Exoplanet Transit Estimator and Visualizer')
st.write('Adjust the sliders below to explore exoplanet and stellar properties, estimate Gaia magnitude, and visualize potential transits.')

# Sliders for input
st.sidebar.header('Input Parameters')
planet_r = st.sidebar.slider('Planet Radius (RE)', min_value=0.1, max_value=30.0, value=1.0, step=0.1)
planet_m = st.sidebar.slider('Planet Mass (ME)', min_value=0.01, max_value=1000.0, value=1.0, step=0.1)
star_r = st.sidebar.slider('Stellar Radius (RSun)', min_value=0.1, max_value=10.0, value=1.0, step=0.05)
star_m = st.sidebar.slider('Stellar Mass (MSun)', min_value=0.1, max_value=5.0, value=1.0, step=0.05)

# Estimation and suggestions
estimated_mag = estimate_gaia_magnitude(planet_r, planet_m, star_r, star_m)

st.subheader('Estimation Results')
if isinstance(estimated_mag, str):
    st.error(f"Estimated Gaia Magnitude: {estimated_mag}")
    plot_title_mag = estimated_mag
    suggested_telescopes = []
else:
    st.success(f"Estimated Gaia Magnitude: {estimated_mag:.2f}")
    plot_title_mag = f"{estimated_mag:.2f}"
    suggested_telescopes = suggest_telescopes(estimated_mag)

if suggested_telescopes:
    st.info(f"Suggested Telescopes for observation: **{', '.join(suggested_telescopes)}**")
elif not isinstance(estimated_mag, str):
    st.warning("No telescope suggestions available for this magnitude.")

# --- Plotting logic ---
st.subheader('Interactive Planet-Star Transit Visualization')

fig, ax = plt.subplots(figsize=(10, 10))

RE_TO_RSUN_FACTOR = 109.2
planet_r_in_rsun = planet_r / RE_TO_RSUN_FACTOR

star_circle = patches.Circle((0, 0), star_r, color='yellow', label='Star', zorder=0)
ax.add_patch(star_circle)

planet_x_position = -star_r + planet_r_in_rsun
planet_circle = patches.Circle((planet_x_position, 0), planet_r_in_rsun, color='black', label='Planet', zorder=1)
ax.add_patch(planet_circle)

margin = 0.2 * star_r
ax.set_xlim(-(star_r + margin), (star_r + margin))
ax.set_ylim(-(star_r + margin), (star_r + margin))
ax.set_aspect('equal', adjustable='box')

# Corrected f-string for title
ax.set_title(f'Planet Radius: {planet_r:.2f} RE, Stellar Radius: {star_r:.2f} RSun\nEstimated Gaia Magnitude: {plot_title_mag}')
ax.set_xlabel('X-position')
ax.set_ylabel('Y-position')
ax.axis('off')

st.pyplot(fig)

st.markdown("""### How to run this application:
1. Save the above code as `app.py` in a directory.
2. Make sure `gaia_mag_random_forest_model.pkl` and `mag_data_for_streamlit.csv` are in the same directory.
3. Open your terminal, navigate to that directory, and run: `streamlit run app.py`""")
st.write("You may need to install Streamlit: `pip install streamlit`")

