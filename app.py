"""
Stunting Classification Web App - Premium Edition v2.0
=======================================================
Aplikasi Streamlit untuk prediksi status stunting anak
dengan fitur Dashboard, Gauge Chart, Animasi Lottie, dan Riwayat Prediksi.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
from streamlit_lottie import st_lottie
import time

# ======================= Lottie Animations =======================
@st.cache_data
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Lottie Animation URLs
LOTTIE_LOADING = "https://assets5.lottiefiles.com/packages/lf20_p8bfn5to.json"
LOTTIE_SUCCESS = "https://assets4.lottiefiles.com/packages/lf20_jbrw3hcz.json"
LOTTIE_CHILD = "https://assets9.lottiefiles.com/packages/lf20_tll0j4bb.json"
LOTTIE_HEALTH = "https://assets2.lottiefiles.com/packages/lf20_5njp3vgg.json"
LOTTIE_CHART = "https://assets7.lottiefiles.com/packages/lf20_kgyknlcb.json"

# ======================= Page Configuration =======================
st.set_page_config(
    page_title="Stunting Classification AI",
    page_icon="🧒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================= Initialize Session State =======================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ======================= Premium Custom CSS =======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Premium Header */
    .premium-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .premium-header h1 { color: #ffffff; font-size: 2.5rem; font-weight: 800; margin-bottom: 10px; text-shadow: 0 2px 10px rgba(0,0,0,0.3); }
    .premium-header p { color: #ffffff; font-size: 1rem; font-weight: 400; }
    .premium-header .badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.25);
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 15px;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    .glass-card h3 { color: #ffffff; font-weight: 600; margin-bottom: 15px; }
    .glass-card p { color: #e0e0e0; }
    
    /* Result Cards */
    .result-card {
        border-radius: 24px;
        padding: 35px;
        text-align: center;
        margin: 15px 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.15) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .result-normal { background: linear-gradient(135deg, #00796b 0%, #388e3c 100%); box-shadow: 0 20px 60px rgba(0, 121, 107, 0.4); }
    .result-tall { background: linear-gradient(135deg, #1976d2 0%, #0288d1 100%); box-shadow: 0 20px 60px rgba(25, 118, 210, 0.4); }
    .result-stunted { background: linear-gradient(135deg, #c2185b 0%, #d32f2f 100%); box-shadow: 0 20px 60px rgba(194, 24, 91, 0.4); }
    .result-severely-stunted { background: linear-gradient(135deg, #b71c1c 0%, #c62828 100%); box-shadow: 0 20px 60px rgba(183, 28, 28, 0.4); }
    
    .result-card h2 { color: #ffffff !important; font-size: 1.3rem; font-weight: 700; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 2px; text-shadow: 0 2px 8px rgba(0,0,0,0.5); }
    .result-card .emoji { font-size: 4rem; margin: 15px 0; }
    .result-card .status { color: #ffffff !important; font-size: 1.8rem; font-weight: 700; margin: 10px 0; text-shadow: 0 2px 8px rgba(0,0,0,0.5); }
    .result-card .confidence { color: #ffffff !important; font-size: 1.1rem; font-weight: 600; text-shadow: 0 1px 4px rgba(0,0,0,0.4); }
    
    /* Stats Cards */
    .stats-card {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .stats-card .value { color: #64ffda; font-size: 2rem; font-weight: 700; }
    .stats-card .label { color: #e0e0e0; font-size: 0.9rem; margin-top: 5px; font-weight: 500; }
    
    /* Info Box */
    .info-box {
        background: rgba(102, 126, 234, 0.25);
        border-left: 4px solid #7c4dff;
        border-radius: 0 16px 16px 0;
        padding: 20px;
        margin: 15px 0;
    }
    
    .info-box h4 { color: #bb86fc; font-weight: 600; margin-bottom: 10px; font-size: 1.1rem; }
    .info-box p { color: #e0e0e0; line-height: 1.6; }
    
    /* Tips List */
    .tips-list { background: rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 20px; margin-top: 15px; }
    .tips-list li { color: #e0e0e0; padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.1); font-size: 0.95rem; }
    .tips-list li:last-child { border-bottom: none; }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 16px 35px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: #e0e0e0 !important; }
    section[data-testid="stSidebar"] label { color: #ffffff !important; font-weight: 500 !important; }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    [data-testid="stMetric"] label { color: #ffffff !important; font-weight: 600 !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #64ffda !important; font-weight: 700 !important; }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #e0e0e0 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #e0e0e0 !important;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
    }
    
    /* History Item */
    .history-item {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Labels and inputs */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    .stSlider [data-baseweb="slider"] { background: rgba(255,255,255,0.2) !important; }
    
    /* Main content text - Force all text white */
    .stMarkdown p { color: #e0e0e0 !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #ffffff !important; }
    .stMarkdown li { color: #e0e0e0 !important; }
    .stMarkdown span { color: #e0e0e0 !important; }
    
    /* Force all text in main area */
    .main .block-container * { color: #e0e0e0; }
    .main .block-container h1, .main .block-container h2, .main .block-container h3 { color: #ffffff !important; }
    
    /* Selectbox and input text */
    .stSelectbox > div > div { color: #ffffff !important; }
    .stSelectbox [data-baseweb="select"] { color: #1a1a2e !important; background: rgba(255,255,255,0.95) !important; }
    .stSelectbox [data-baseweb="select"] * { color: #1a1a2e !important; }
    .stSelectbox [data-baseweb="select"] span { color: #1a1a2e !important; }
    .stSelectbox [data-baseweb="select"] div { color: #1a1a2e !important; }
    .stSelectbox [data-baseweb="select"] svg { fill: #1a1a2e !important; }
    .stSelectbox div[data-testid="stSelectbox"] div[class*="ValueContainer"] { color: #1a1a2e !important; }
    .stSelectbox div[class*="singleValue"] { color: #1a1a2e !important; }
    
    /* Force dark text on select input */
    [data-baseweb="select"] [class*="value"] { color: #1a1a2e !important; }
    [data-baseweb="select"] [class*="Value"] { color: #1a1a2e !important; }
    [data-baseweb="select"] [class*="placeholder"] { color: #666666 !important; }
    [data-baseweb="select"] input { color: #1a1a2e !important; }
    
    /* Dropdown menu items */
    [data-baseweb="popover"] { background: #ffffff !important; }
    [data-baseweb="popover"] * { color: #1a1a2e !important; }
    [data-baseweb="menu"] { background: #ffffff !important; }
    [data-baseweb="menu"] * { color: #1a1a2e !important; }
    [data-baseweb="menu"] li { color: #1a1a2e !important; }
    [data-baseweb="menu"] li:hover { background: rgba(102, 126, 234, 0.2) !important; }
    [role="listbox"] li { color: #1a1a2e !important; background: #ffffff !important; }
    [role="option"] { color: #1a1a2e !important; }
    
    /* Slider text values */
    .stSlider > div > div > div { color: #ffffff !important; }
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #ffffff !important; }
    
    /* Widget labels */
    .stWidgetLabel { color: #ffffff !important; }
    [data-testid="stWidgetLabel"] { color: #ffffff !important; }
    [data-testid="stWidgetLabel"] p { color: #ffffff !important; }
    
    /* Expander */
    .streamlit-expanderHeader { color: #ffffff !important; }
    
    /* Caption and small text */
    .stCaption { color: #b0b0b0 !important; }
    small { color: #b0b0b0 !important; }
    
    /* Dataframe text */
    .stDataFrame { color: #ffffff !important; }
    
    /* Help text */
    [data-testid="stTooltipHoverTarget"] { color: #ffffff !important; }
    
    /* Footer */
    .premium-footer {
        text-align: center;
        padding: 30px;
        margin-top: 40px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .premium-footer p { color: #b0b0b0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ======================= Load Models =======================
@st.cache_resource
def load_models():
    MODEL_PATH = 'models/'
    try:
        return {
            'knn': joblib.load(f'{MODEL_PATH}knn_model.joblib'),
            'rf': joblib.load(f'{MODEL_PATH}rf_model.joblib'),
            'scaler': joblib.load(f'{MODEL_PATH}scaler.joblib'),
            'le_gender': joblib.load(f'{MODEL_PATH}label_encoder_gender.joblib'),
            'le_stunting': joblib.load(f'{MODEL_PATH}label_encoder_stunting.joblib'),
            'model_info': joblib.load(f'{MODEL_PATH}model_info.joblib'),
            'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('stunting_wasting_dataset.csv')
    except:
        return None

# ======================= Helper Functions =======================
def get_result_class(prediction):
    return {
        'Normal': 'result-normal',
        'Tall': 'result-tall',
        'Stunted': 'result-stunted',
        'Severely Stunted': 'result-severely-stunted'
    }.get(prediction, 'result-normal')

def get_recommendation(prediction):
    recommendations = {
        'Normal': {'icon': '✅', 'emoji': '😊', 'status': 'Status Normal', 'color': '#00b09b',
            'message': 'Anak memiliki pertumbuhan yang normal dan sehat.',
            'tips': ['✓ Pertahankan pola makan sehat', '✓ ASI eksklusif hingga 6 bulan', '✓ MPASI bervariasi', '✓ Pantau pertumbuhan rutin', '✓ Imunisasi lengkap']},
        'Tall': {'icon': '📏', 'emoji': '🌟', 'status': 'Tinggi di Atas Rata-rata', 'color': '#4facfe',
            'message': 'Anak memiliki tinggi di atas rata-rata.',
            'tips': ['✓ Jaga gizi seimbang', '✓ Konsultasi dokter', '✓ Pantau berat badan', '✓ Aktivitas fisik cukup']},
        'Stunted': {'icon': '⚠️', 'emoji': '😟', 'status': 'Stunting (Pendek)', 'color': '#fa709a',
            'message': 'Anak terindikasi stunting. Perlu perhatian khusus.',
            'tips': ['⚡ Konsultasi dokter/ahli gizi', '⚡ Tingkatkan protein', '⚡ Makanan kaya zat besi', '⚡ Kebersihan lingkungan', '⚡ Pantau intensif']},
        'Severely Stunted': {'icon': '🚨', 'emoji': '😢', 'status': 'Severely Stunted', 'color': '#ff416c',
            'message': 'Stunting berat. SEGERA lakukan penanganan!',
            'tips': ['🚨 SEGERA ke faskes', '🚨 Dokter spesialis anak', '🚨 Program intervensi gizi', '🚨 Tingkatkan kalori', '🚨 Cek infeksi']}
    }
    return recommendations.get(prediction, recommendations['Normal'])

def create_gauge_chart(confidence, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': 'white'}},
        number={'font': {'size': 40, 'color': 'white'}, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
            'bar': {'color': '#667eea'},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 65, 108, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(250, 112, 154, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(0, 176, 155, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#00f2fe', 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# ======================= Main App =======================
def main():
    # Header
    st.markdown("""
    <div class="premium-header">
        <h1>🧒 Stunting Classification AI</h1>
        <p>Sistem Prediksi Status Stunting Anak Berbasis Machine Learning</p>
        <div class="badge">✨ v2.0 - Dashboard | Gauge Chart | History</div>
    </div>
    """, unsafe_allow_html=True)
    
    models = load_models()
    df = load_dataset()
    
    if not models['loaded']:
        st.error(f"⚠️ Model belum tersedia! Error: {models.get('error', 'Unknown')}")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 style="text-align:center;">⚙️ Pengaturan</h2>', unsafe_allow_html=True)
        
        model_choice = st.selectbox("🤖 Pilih Model", ['🌲 Random Forest', '📍 KNN', '⚡ Bandingkan'], index=0)
        
        st.markdown("---")
        st.markdown('<h3>📊 Performa Model</h3>', unsafe_allow_html=True)
        
        model_info = models['model_info']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("KNN", f"{model_info['knn_metrics']['accuracy']*100:.1f}%")
        with col2:
            st.metric("RF", f"{model_info['rf_metrics']['accuracy']*100:.1f}%")
        
        st.markdown("---")
        if len(st.session_state.prediction_history) > 0 and st.button("🗑️ Hapus Riwayat"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Dashboard", "📜 Riwayat"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="glass-card"><h3>📝 Input Data Anak</h3></div>', unsafe_allow_html=True)
            
            jenis_kelamin = st.selectbox("👤 Jenis Kelamin", ['Laki-laki', 'Perempuan'])
            umur = st.slider("📅 Umur (bulan)", 0, 60, 12)
            tinggi_badan = st.slider("📏 Tinggi Badan (cm)", 40.0, 130.0, 75.0, 0.5)
            berat_badan = st.slider("⚖️ Berat Badan (kg)", 2.0, 30.0, 9.0, 0.1)
            
            predict_btn = st.button("🔮 Prediksi Status Stunting", use_container_width=True)
        
        with col2:
            if predict_btn:
                gender_encoded = models['le_gender'].transform([jenis_kelamin])[0]
                input_data = pd.DataFrame({
                    'Jenis Kelamin': [gender_encoded],
                    'Umur (bulan)': [umur],
                    'Tinggi Badan (cm)': [tinggi_badan],
                    'Berat Badan (kg)': [berat_badan]
                })
                input_scaled = models['scaler'].transform(input_data)
                
                pred_knn = models['knn'].predict(input_scaled)[0]
                pred_rf = models['rf'].predict(input_scaled)[0]
                prob_knn = models['knn'].predict_proba(input_scaled)[0]
                prob_rf = models['rf'].predict_proba(input_scaled)[0]
                
                result_knn = models['le_stunting'].inverse_transform([pred_knn])[0]
                result_rf = models['le_stunting'].inverse_transform([pred_rf])[0]
                
                if 'Random Forest' in model_choice:
                    result, prob, model_name = result_rf, prob_rf, "Random Forest"
                elif 'KNN' in model_choice:
                    result, prob, model_name = result_knn, prob_knn, "KNN"
                else:
                    result, prob, model_name = result_rf, prob_rf, "Random Forest"
                
                # Save to history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'gender': jenis_kelamin,
                    'age': umur,
                    'height': tinggi_badan,
                    'weight': berat_badan,
                    'result': result,
                    'confidence': max(prob) * 100,
                    'model': model_name
                })
                
                rec = get_recommendation(result)
                
                # Success Animation
                lottie_success = load_lottie_url(LOTTIE_SUCCESS)
                if lottie_success:
                    st_lottie(lottie_success, height=100, key="success_anim")
                
                st.markdown(f"""
                <div class="result-card {get_result_class(result)}">
                    <h2>🎯 {model_name}</h2>
                    <div class="emoji">{rec['emoji']}</div>
                    <div class="status">{result}</div>
                    <div class="confidence">Confidence: {max(prob)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge Chart
                st.plotly_chart(create_gauge_chart(max(prob)*100, "Confidence Score"), use_container_width=True)
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 40px;">
                    <h3 style="color: rgba(255,255,255,0.6);">🔮 Hasil Prediksi</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Lottie Animation Placeholder
                lottie_child = load_lottie_url(LOTTIE_CHILD)
                if lottie_child:
                    st_lottie(lottie_child, height=200, key="child_anim")
                
                st.markdown("""
                <p style="text-align: center; color: rgba(255,255,255,0.4);">Masukkan data anak dan klik tombol prediksi</p>
                """, unsafe_allow_html=True)
        
        # Recommendations
        if predict_btn:
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="glass-card">
                    <h3>💡 Rekomendasi</h3>
                    <div class="info-box">
                        <h4>{rec['status']}</h4>
                        <p>{rec['message']}</p>
                    </div>
                    <div class="tips-list">
                        <ul style="list-style: none; padding: 0;">
                            {''.join([f'<li>{tip}</li>' for tip in rec['tips']])}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="glass-card"><h3>📈 Perbandingan Probabilitas</h3></div>', unsafe_allow_html=True)
                classes = models['le_stunting'].classes_
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Random Forest', x=classes, y=prob_rf*100, marker_color='#667eea', text=[f'{v:.1f}%' for v in prob_rf*100], textposition='outside', textfont=dict(color='white')))
                fig.add_trace(go.Bar(name='KNN', x=classes, y=prob_knn*100, marker_color='#00f2fe', text=[f'{v:.1f}%' for v in prob_knn*100], textposition='outside', textfont=dict(color='white')))
                fig.update_layout(
                    barmode='group', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff', size=12),
                    height=300, 
                    margin=dict(l=20, r=20, t=40, b=40),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.2)', title=dict(text='%', font=dict(color='#ffffff')), tickfont=dict(color='#ffffff')),
                    xaxis=dict(tickfont=dict(color='#ffffff')),
                    legend=dict(font=dict(color='#ffffff', size=12))
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="glass-card"><h3>📊 Dashboard Statistik Dataset</h3></div>', unsafe_allow_html=True)
        
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'<div class="stats-card"><div class="value">{len(df):,}</div><div class="label">Total Data</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stats-card"><div class="value">{len(df.columns)}</div><div class="label">Fitur</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="stats-card"><div class="value">{df["Stunting"].nunique()}</div><div class="label">Kelas Target</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stats-card"><div class="value">{df.isnull().sum().sum()}</div><div class="label">Missing Values</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="glass-card"><h3>📊 Distribusi Stunting</h3></div>', unsafe_allow_html=True)
                stunting_counts = df['Stunting'].value_counts()
                fig = px.pie(values=stunting_counts.values, names=stunting_counts.index,
                    color_discrete_sequence=['#00b09b', '#4facfe', '#fa709a', '#ff416c'])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font=dict(color='#ffffff', size=12), 
                    height=350,
                    legend=dict(font=dict(color='#ffffff', size=12))
                )
                fig.update_traces(textfont=dict(color='#ffffff', size=12))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown('<div class="glass-card"><h3>📊 Distribusi Jenis Kelamin</h3></div>', unsafe_allow_html=True)
                gender_counts = df['Jenis Kelamin'].value_counts()
                fig = px.bar(x=gender_counts.index, y=gender_counts.values,
                    color=gender_counts.index, color_discrete_sequence=['#667eea', '#00f2fe'],
                    text=gender_counts.values)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff', size=12), 
                    height=350, 
                    showlegend=False,
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#ffffff')),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#ffffff'))
                )
                fig.update_traces(textfont=dict(color='#ffffff', size=12), textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dataset tidak ditemukan")
    
    with tab3:
        st.markdown('<div class="glass-card"><h3>📜 Riwayat Prediksi</h3></div>', unsafe_allow_html=True)
        
        if len(st.session_state.prediction_history) > 0:
            for i, pred in enumerate(reversed(st.session_state.prediction_history[-10:])):
                color = get_recommendation(pred['result'])['color']
                st.markdown(f"""
                <div class="history-item" style="border-left-color: {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="color: {color}; font-weight: 700; font-size: 1.2rem;">{pred['result']}</span>
                            <span style="color: rgba(255,255,255,0.5); margin-left: 10px;">({pred['confidence']:.1f}%)</span>
                        </div>
                        <span style="color: rgba(255,255,255,0.4);">{pred['timestamp']}</span>
                    </div>
                    <div style="color: rgba(255,255,255,0.7); margin-top: 8px; font-size: 0.9rem;">
                        {pred['gender']} • {pred['age']} bulan • {pred['height']} cm • {pred['weight']} kg • Model: {pred['model']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: rgba(255,255,255,0.5);">
                <div style="font-size: 3rem;">📭</div>
                <p>Belum ada riwayat prediksi</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="premium-footer">
        <p>🎓 Tugas Besar Praktikum Machine Learning</p>
        <p>Made with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
