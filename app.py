import streamlit as st
import pandas as pd
import sys
if 'data' in sys.modules:
    del sys.modules['data']
from data import get_data
from google import genai
from PIL import Image, ExifTags
import io
import os
from fpdf import FPDF

if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

st.set_page_config(page_title="MPLAD Integrity Engine", layout="wide")

st.title("MPLAD Integrity Engine")

if "notice_id" in st.query_params:
    notice_id = st.query_params["notice_id"]
    st.success(f"✅ **Verification Successful:** Notice `{notice_id}` is a valid document issued by the District Audit Office and securely logged in the e-SAKSHI ledger.")

# Load data (cache removed temporarily for debugging)
def load_data():
    data = get_data()
    return data

df = load_data()
print("Loaded Columns:", list(df.columns))

def get_decimal_from_dms(dms, ref):
    degrees, minutes, seconds = dms[0], dms[1], dms[2]
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_exif_with_gps(image_file):
    try:
        img = Image.open(image_file)
        exif_data = img._getexif()
        if not exif_data:
            return None, None, "No EXIF metadata found."
        
        gps_info = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = ExifTags.GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
        
        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
            lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
            return lat, lon, "GPS Data Extracted Successfully"
        
        return None, None, "Image has EXIF but no GPS coordinates."
    except Exception as e:
        return None, None, f"Extraction failed: {e}"

def verify_site_photo_with_ai(image_file, project_context):
    """Uses Gemini Vision to verify if the photo shows real construction work."""
    client = genai.Client()
    img = Image.open(image_file)
    
    prompt = f"""
    Analyze this photo submitted by a contractor for MPLADS project: '{project_context}'.
    1. Does this photo depict genuine on-ground physical infrastructure/construction work?
    2. Is there any sign of this being a digital stock image, indoor mockup, or unrelated object?
    3. Return a one-line verdict: [VERIFIED REAL SITE] or [SUSPECT GHOST ASSET].
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[img, prompt]
    )
    return response.text

@st.cache_data(show_spinner=False)
def generate_ai_forensic_brief(mp_name, constituency, state, prog_pct, util_pct, vendor_name, vendor_reps, rules_str):
    client = genai.Client() 
    prompt = f"""
    You are a CAG Forensic Auditor. Analyze this flagged MPLADS project:
    - MP: {mp_name} ({constituency}, {state})
    - Physical Progress: {prog_pct}% | Funds Disbursed: {util_pct}%
    - Vendor: {vendor_name} (Repeated {vendor_reps} times)
    - Anomalies: {rules_str}
    
    Provide exactly 3 concise bullet points:
    1. Primary Risk Vector (Why it was flagged)
    2. Potential Scheme Violation (e.g., Ghost asset, milestone mismatch)
    3. Actionable Recommendation for District Collector
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return response.text

def create_show_cause_notice(mp_name, constituency, vendor, reason):
    import datetime
    import random
    import string
    import qrcode
    import tempfile
    import os
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "GOVERNMENT OF INDIA - DISTRICT AUDIT OFFICE", ln=True, align="C")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "SHOW CAUSE NOTICE: MPLADS FUND DISCREPANCY", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Constituency: {constituency} | Hon'ble MP: {mp_name}", ln=True)
    pdf.cell(0, 6, f"Contractor / Implementing Agency: {vendor}", ln=True)
    pdf.ln(4)
    
    body = (
        f"Notice is hereby served under e-SAKSHI Fund Monitoring Guidelines. An automated CAG "
        f"audit flagged this project under violation code(s): {reason}.\n\n"
        f"You are instructed to provide physical milestone verification documents and geo-tagged "
        f"evidence within 7 working days. Failure to comply will lead to immediate escrow account freeze."
    )
    pdf.multi_cell(0, 6, body)
    
    pdf.ln(10)
    
    # Generate Reference ID
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    rand_suffix = ''.join(random.choices(string.digits, k=3))
    const_code = str(constituency).replace(" ", "").upper()[:5] if constituency else "UNKNOWN"
    ref_id = f"Notice ID: MPLADS-{const_code}-{date_str}-{rand_suffix}"
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, ref_id, ln=True)
    
    # Generate QR Code
    qr = qrcode.QRCode(box_size=4, border=1)
    
    # Hardcoded Streamlit Cloud URL for the QR code
    verification_url = f"https://mplad-integrity-engine-lujvjnrykbefiwdxjk5cg5.streamlit.app/?notice_id={ref_id.split(': ')[-1]}"
    qr.add_data(verification_url)
    
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save the QR code temporarily and embed it safely
    tmp_fd, tmp_name = tempfile.mkstemp(suffix='.png')
    os.close(tmp_fd) # Close file descriptor so FPDF can read it on Windows
    
    img.save(tmp_name)
    pdf.image(tmp_name, w=25)
    
    # Cleanup the temp file
    try:
        os.remove(tmp_name)
    except:
        pass
        
    return bytes(pdf.output())

# Sidebar filters
st.sidebar.header("Filters")

# House filter
all_houses = ["Lok Sabha", "Rajya Sabha"]
selected_houses = st.sidebar.multiselect("Select House(s)", all_houses, default=all_houses)

# State filter
all_states = sorted(df['state'].unique().tolist())
selected_states = st.sidebar.multiselect("Select State(s)", all_states, default=all_states)

# Risk Level filter
all_risks = ["High", "Medium", "Low"]
selected_risks = st.sidebar.multiselect("Select Risk Level(s)", all_risks, default=["High", "Medium"])

# Apply filters
filtered_df = df[
    (df['house'].isin(selected_houses)) &
    (df['state'].isin(selected_states)) &
    (df['risk_level'].isin(selected_risks))
].copy()

# Top Metrics
total_flagged = len(filtered_df)
high_risk_count = len(filtered_df[filtered_df['risk_level'] == 'High'])
# Total Funds at Risk in Cr (sum of fund_sanctioned for high/medium risk projects)
funds_at_risk = filtered_df[filtered_df['risk_level'].isin(['High', 'Medium'])]['fund_sanctioned'].sum()
funds_at_risk_cr = funds_at_risk / 1e7  # convert to Cr

col1, col2, col3 = st.columns(3)
col1.metric("Total Flagged", total_flagged)
col2.metric("High Risk Count", high_risk_count)
col3.metric("Total Funds at Risk (Cr)", f"₹ {funds_at_risk_cr:,.2f} Cr")

st.markdown("---")

# Flagged Projects Table
st.subheader("Flagged Projects Overview")

# Sort by risk (High > Medium > Low) and anomaly_score (lower is worse)
sort_map = {"High": 1, "Medium": 2, "Low": 3}
filtered_df['risk_sort'] = filtered_df['risk_level'].map(sort_map)
sorted_df = filtered_df.sort_values(by=['risk_sort', 'anomaly_score'], ascending=[True, True])

# Add physical progress to your display columns
display_cols = ['mp_name', 'house', 'state', 'constituency', 'physical_progress_percent', 'risk_level', 'anomaly_score', 'rules_str']

st.dataframe(
    sorted_df[display_cols], 
    column_config={
        "mp_name": "MP Name",
        "house": "House",
        "state": "State",
        "constituency": "Constituency",
        "physical_progress_percent": st.column_config.ProgressColumn(
            "Physical Progress", 
            help="Reported ground progress",
            format="%.1f%%", 
            min_value=0, 
            max_value=100
        ),
        "risk_level": st.column_config.TextColumn("Risk Level"),
        "anomaly_score": st.column_config.NumberColumn("Risk Score", format="%.3f"),
        "rules_str": "Rules Triggered"
    },
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# Bar Chart: Flagged count by state
st.subheader("Flagged Count by State")
if not filtered_df.empty:
    state_counts = filtered_df['state'].value_counts()
    st.bar_chart(state_counts)
else:
    st.write("No data available for the selected filters.")

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📈 Anomaly & Fraud Matrix", 
    "⏱️ Inefficiency & Sanction Delays", 
    "📷 Ghost Asset Geo-Verification"
])

with tab1:
    st.subheader("Project Deep Dive & AI Forensic Brief")
    if not sorted_df.empty:
        # Create a nice label for the selectbox
        project_options = sorted_df['sr_no'].tolist()
        
        def format_project(sr_no):
            row = sorted_df[sorted_df['sr_no'] == sr_no].iloc[0]
            return f"{row['mp_name']} ({row['constituency']}, {row['state']}) - Risk: {row['risk_level']}"
            
        selected_sr_no = st.selectbox("Select a flagged project", project_options, format_func=format_project)
        
        if selected_sr_no:
            proj_data = sorted_df[sorted_df['sr_no'] == selected_sr_no].iloc[0]
            
            st.markdown(f"**MP Name:** {proj_data['mp_name']} | **Constituency:** {proj_data['constituency']} | **State:** {proj_data['state']}")
            
            # Summary text
            util_pct = round(proj_data['fund_utilization_ratio'] * 100, 1)
            prog_pct = round(proj_data['physical_progress_percent'], 1)
            vendor_reps = proj_data['vendor_repetition_count']
            
            st.info(f"**Raw Data Match:** Flagged due to {util_pct}% utilization vs {prog_pct}% progress. Vendor '{proj_data['vendor_name']}' used {vendor_reps} times.")

            st.markdown("### 🤖 AI Forensic Audit Brief")
            with st.spinner("Generating automated CAG assessment..."):
                try:
                    brief_text = generate_ai_forensic_brief(
                        proj_data['mp_name'], 
                        proj_data['constituency'], 
                        proj_data['state'], 
                        prog_pct, 
                        util_pct, 
                        proj_data['vendor_name'], 
                        vendor_reps, 
                        proj_data['rules_str']
                    )
                    st.warning(brief_text)
                except Exception as e:
                    st.error(f"AI Audit unavailable. Ensure GEMINI_API_KEY is configured. Error: {e}")
            
            # Simple bar chart of the 4 features
            st.write("**Raw Feature Values (Proxy for Feature Importance)**")
            features_to_plot = {
                'Utilization Ratio (x100 for scale)': float(proj_data['fund_utilization_ratio'] * 100),
                'Progress Gap (x100 for scale)': float(proj_data['progress_gap'] * 100),
                'Months Since Sanction': float(proj_data['months_since_sanction']),
                'Vendor Repetitions': float(proj_data['vendor_repetition_count'])
            }
            
            feat_df = pd.DataFrame({
                'Feature': list(features_to_plot.keys()),
                'Value': list(features_to_plot.values())
            }).set_index('Feature')
            
            st.bar_chart(feat_df)
            
            # Add Show Cause Notice button:
            notice_bytes = create_show_cause_notice(
                proj_data['mp_name'], 
                proj_data['constituency'], 
                proj_data['vendor_name'], 
                proj_data['rules_str']
            )
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.download_button(
                    label="📄 Generate Show-Cause Notice (PDF)",
                    data=notice_bytes,
                    file_name=f"Notice_{proj_data['constituency']}.pdf",
                    mime="application/pdf"
                )
            with col_btn2:
                if st.button("🚨 Freeze Escrow Payment Milestone"):
                    st.error(f"Milestone funds for {proj_data['vendor_name']} successfully locked in e-SAKSHI ledger.")

with tab2:
    st.subheader("Administrative Bottlenecks & Idle Funds")
    
    # Filter on filtered_df so sidebar state selections apply here too
    delayed_df = filtered_df[filtered_df['days_pending_sanction'] > 60].copy()
    
    if not delayed_df.empty:
        # Convert unspent balance to Crores for clean display
        delayed_df['unspent_balance_cr'] = delayed_df['unspent_balance'] / 1e7
        
        display_delay_cols = ['mp_name', 'state', 'constituency', 'days_pending_sanction', 'unspent_balance_cr', 'rules_str']
        
        st.dataframe(
            delayed_df[display_delay_cols],
            column_config={
                "mp_name": "MP Name",
                "state": "State",
                "constituency": "Constituency",
                "days_pending_sanction": st.column_config.NumberColumn(
                    "Sanction Pending (Days)",
                    help="Days elapsed since MP recommendation without administrative sanction",
                    format="%d days"
                ),
                "unspent_balance_cr": st.column_config.NumberColumn(
                    "Idle Funds (Cr)",
                    help="Unspent allocated budget sitting in treasury",
                    format="₹ %.2f Cr"
                ),
                "rules_str": "Bottleneck Classification"
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No critical administrative delays detected in the selected states.")

with tab3:
    st.subheader("Site Geotag & Photo Verification")
    uploaded_photo = st.file_uploader("Upload Contractor Progress Photo", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_photo:
        col_img, col_map = st.columns(2)
        
        with col_img:
            st.image(uploaded_photo, use_container_width=True, caption="Uploaded Site Photo")
            
            uploaded_photo.seek(0)  # reset pointer before first read
            lat, lon, msg = extract_exif_with_gps(uploaded_photo)
            st.write(f"**Metadata Status:** {msg}")
            
            if st.button("Run AI Vision Authenticity Scan"):
                with st.spinner("Analyzing structural authenticity..."):
                    uploaded_photo.seek(0)  # reset pointer again before second read
                    verdict = verify_site_photo_with_ai(uploaded_photo, "MPLADS Construction Site")
                    st.info(verdict)
                    
        with col_map:
            if lat is not None and lon is not None:
                try:
                    lat_val = float(lat)
                    lon_val = float(lon)
                    st.write("**Detected GPS Asset Location:**")
                    map_df = pd.DataFrame({
                        "latitude": [lat_val],
                        "longitude": [lon_val]
                    })
                    st.map(map_df, zoom=14, use_container_width=True)
                except (ValueError, TypeError) as e:
                    st.warning(f"⚠️ GPS data found but could not be parsed: {e}")
            else:
                st.warning("⚠️ No geotag detected. High probability of stock/fraudulent image.")

