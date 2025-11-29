# crowdguard_dashboard.py
import streamlit as st
from pathlib import Path
import time
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="CrowdGuard Live Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DASHBOARD_DATA_DIR = ROOT / "dashboard_data"
FRAME_PATH = DASHBOARD_DATA_DIR / "latest_frame.jpg"
ALERT_PATH = DASHBOARD_DATA_DIR / "latest_alert.txt"

# --- UI Components ---
st.title("🛡️ CrowdGuard Live Monitoring")

# Define placeholders for the UI elements
st.header("Live Video Feed")
image_placeholder = st.empty()

st.header("AI Analysis")
alert_placeholder = st.empty()
suggestion_placeholder = st.empty()

# --- Main Loop ---
while True:
    try:
        # --- Display Video Frame ---
        if FRAME_PATH.exists():
            with open(FRAME_PATH, "rb") as f:
                # Convert image to base64 to avoid disk caching issues
                img_bytes = base64.b64encode(f.read()).decode()
                image_placeholder.image(
                    f"data:image/jpeg;base64,{img_bytes}",
                    use_container_width=True,
                    caption=f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(FRAME_PATH.stat().st_mtime))}"
                )

        # --- Display AI Alert ---
        if ALERT_PATH.exists():
            with open(ALERT_PATH, "r", encoding="utf-8") as f:
                alert_text = f.read()
            
            alert_content = ""
            suggestion_content = ""

            if "Alert:" in alert_text and "Suggestion:" in alert_text:
                # Split the alert text by newline characters
                lines = alert_text.split('\n')
                # Ensure there are at least two lines before attempting to access lines[1]
                if len(lines) >= 2:
                    alert_content = lines[0].replace("Alert:", "").strip()
                    suggestion_content = lines[1].replace("Suggestion:", "").strip()
                else:
                    # Handle cases where the expected multi-line format is not present
                    alert_content = "Malformed AI response detected."
                    suggestion_content = "Check AI output format."

            # Clear previous state
            alert_placeholder.empty()
            suggestion_placeholder.empty()

            if "Standby" in alert_text:
                alert_placeholder.info(f"**Status:** {alert_text}", icon="⏳")
            elif "Error" in alert_text:
                alert_placeholder.error(f"**Status:** {alert_text}", icon="🔥")
            elif alert_content:
                alert_placeholder.warning(f"**ALERT:** {alert_content}", icon="⚠️")
                suggestion_placeholder.info(f"**Suggestion:** {suggestion_content}", icon="💡")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    time.sleep(0.5) # Refresh rate (in seconds)