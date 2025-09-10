import streamlit as st
import json
import torch
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import calendar
from pathlib import Path
import io
import base64
import hashlib
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Core ML/AI imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification,
    pipeline, AutoModelForSequenceClassification
)

# Image/OCR processing
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import cv2

# Audio processing
import speech_recognition as sr
import pydub
from pydub import AudioSegment
import tempfile
import os

# Explainable AI
import shap
import lime
from lime.lime_text import LimeTextExplainer

# Knowledge Graph & Drug Interactions
import platform

# Configure Tesseract path for Windows
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Page config
st.set_page_config(
    page_title="🚀 AI Medical Prescription System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for badass styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #fd79a8, #e84393);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #00b894, #00a085);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🚀 NEXT-GEN AI Medical Prescription System</h1><p>Smart • Safe • Reliable • Explainable</p></div>', unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'drug_graph' not in st.session_state:
    st.session_state.drug_graph = None

# 🔥 FEATURE 1: Advanced Drug Data with NER capabilities
@st.cache_data
def load_enhanced_drug_data():
    """Load comprehensive drug database with enhanced features"""
    try:
        with open("data/enhanced_drug_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback enhanced data structure
        return {
            "drugs": [
                {
                    "name": "Paracetamol",
                    "generic_names": ["Acetaminophen"],
                    "drug_class": "Analgesic/Antipyretic",
                    "adult_dosage": "500-1000mg every 4-6 hours",
                    "pediatric_dosage": "10-15mg/kg every 4-6 hours",
                    "max_daily_dose": "4000mg",
                    "side_effects": ["Nausea", "Liver damage (overdose)", "Allergic reactions"],
                    "contraindications": "Severe liver disease, Alcohol dependency",
                    "interactions": ["Warfarin", "Alcohol", "Carbamazepine"],
                    "pregnancy_category": "B",
                    "cost_range": [2, 15],
                    "availability_score": 0.95,
                    "generic_available": True,
                    "brand_names": ["Tylenol", "Calpol", "Panadol"]
                },
                {
                    "name": "Ibuprofen",
                    "generic_names": ["Ibuprofen"],
                    "drug_class": "NSAID",
                    "adult_dosage": "400-800mg every 6-8 hours",
                    "pediatric_dosage": "5-10mg/kg every 6-8 hours",
                    "max_daily_dose": "2400mg",
                    "side_effects": ["Stomach upset", "Increased bleeding risk", "Kidney problems"],
                    "contraindications": "Pregnancy (3rd trimester), Severe heart failure, Active GI bleeding",
                    "interactions": ["Warfarin", "ACE inhibitors", "Lithium"],
                    "pregnancy_category": "C/D",
                    "cost_range": [3, 20],
                    "availability_score": 0.92,
                    "generic_available": True,
                    "brand_names": ["Advil", "Motrin", "Brufen"]
                },
                {
                    "name": "Metformin",
                    "generic_names": ["Metformin HCl"],
                    "drug_class": "Antidiabetic (Biguanide)",
                    "adult_dosage": "500-1000mg twice daily",
                    "pediatric_dosage": "Not recommended <10 years",
                    "max_daily_dose": "2550mg",
                    "side_effects": ["GI upset", "Lactic acidosis (rare)", "Vitamin B12 deficiency"],
                    "contraindications": "Severe kidney disease, Metabolic acidosis",
                    "interactions": ["Alcohol", "Contrast agents", "Diuretics"],
                    "pregnancy_category": "B",
                    "cost_range": [10, 50],
                    "availability_score": 0.88,
                    "generic_available": True,
                    "brand_names": ["Glucophage", "Fortamet"]
                }
            ],
            "drug_alternatives": {
                "Paracetamol": ["Ibuprofen", "Aspirin", "Diclofenac"],
                "Ibuprofen": ["Paracetamol", "Naproxen", "Celecoxib"],
                "Metformin": ["Gliclazide", "Linagliptin", "Empagliflozin"]
            },
            "high_risk_combinations": [
                {
                    "drugs": ["Warfarin", "Aspirin"],
                    "risk_level": "HIGH",
                    "mechanism": "Increased bleeding risk",
                    "recommendation": "Monitor INR closely, consider PPI"
                },
                {
                    "drugs": ["Metformin", "Contrast agents"],
                    "risk_level": "MODERATE",
                    "mechanism": "Risk of lactic acidosis",
                    "recommendation": "Stop metformin 48h before contrast"
                }
            ]
        }

# 🔥 FEATURE 2: Knowledge Graph for Drug Interactions
class DrugInteractionGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.interaction_weights = {}
    
    def build_graph(self, drug_data):
        """Build drug interaction network"""
        # Add nodes (drugs)
        for drug in drug_data['drugs']:
            self.graph.add_node(drug['name'], 
                              drug_class=drug.get('drug_class', ''),
                              risk_score=self._calculate_risk_score(drug))
        
        # Add edges (interactions)
        for combo in drug_data.get('high_risk_combinations', []):
            if len(combo['drugs']) >= 2:
                drug1, drug2 = combo['drugs'][0], combo['drugs'][1]
                weight = {'HIGH': 3, 'MODERATE': 2, 'LOW': 1}.get(combo['risk_level'], 1)
                self.graph.add_edge(drug1, drug2, 
                                  weight=weight,
                                  risk_level=combo['risk_level'],
                                  mechanism=combo['mechanism'])
    
    def _calculate_risk_score(self, drug):
        """Calculate overall risk score for a drug"""
        score = 0
        score += len(drug.get('side_effects', [])) * 0.1
        score += len(drug.get('interactions', [])) * 0.2
        if 'contraindications' in drug:
            score += 0.5
        return min(score, 1.0)
    
    def find_interactions(self, drug_list):
        """Find all interactions in a drug list"""
        interactions = []
        for i, drug1 in enumerate(drug_list):
            for drug2 in drug_list[i+1:]:
                if self.graph.has_edge(drug1, drug2):
                    edge_data = self.graph.get_edge_data(drug1, drug2)
                    interactions.append({
                        'drug1': drug1,
                        'drug2': drug2,
                        'risk_level': edge_data['risk_level'],
                        'mechanism': edge_data['mechanism'],
                        'weight': edge_data['weight']
                    })
        return interactions
    
    def visualize_interactions(self, drug_list):
        """Create interaction network visualization"""
        subgraph = self.graph.subgraph(drug_list)
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(subgraph)
        
        # Color nodes by risk
        node_colors = []
        for node in subgraph.nodes():
            risk_score = subgraph.nodes[node].get('risk_score', 0)
            if risk_score > 0.7:
                node_colors.append('red')
            elif risk_score > 0.4:
                node_colors.append('orange')
            else:
                node_colors.append('green')
        
        nx.draw(subgraph, pos, node_color=node_colors, 
                with_labels=True, font_size=10, font_weight='bold')
        plt.title("Drug Interaction Network")
        return plt.gcf()

# 🔥 FEATURE 3: Advanced NER for Prescription Parsing
class PrescriptionNER:
    def __init__(self):
        try:
            # Try to load a medical NER model
            self.ner_pipeline = pipeline("ner", 
                                       model="d4data/biomedical-ner-all", 
                                       aggregation_strategy="simple")
        except:
            # Fallback to general NER
            self.ner_pipeline = pipeline("ner", 
                                       model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                       aggregation_strategy="simple")
    
    def extract_entities(self, prescription_text):
        """Extract structured entities from prescription text"""
        entities = self.ner_pipeline(prescription_text)
        
        # Pattern-based extraction as fallback
        patterns = {
            'dosage': r'(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?)',
            'frequency': r'(once|twice|thrice|\d+\s*times?)\s*(daily|per day|a day)',
            'duration': r'for\s+(\d+)\s*(days?|weeks?|months?)',
            'route': r'(oral|IV|IM|subcutaneous|topical|inhaled)'
        }
        
        structured_data = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, prescription_text, re.IGNORECASE)
            structured_data[key] = matches
        
        # Combine NER results with pattern matching
        return {
            'ner_entities': entities,
            'structured_data': structured_data,
            'confidence_score': self._calculate_confidence(entities, structured_data)
        }
    
    def _calculate_confidence(self, entities, structured_data):
        """Calculate confidence in extraction"""
        base_score = 0.5
        if entities:
            base_score += 0.3
        if any(structured_data.values()):
            base_score += 0.2
        return min(base_score, 1.0)

# 🔥 FEATURE 4: Smart Dosing Engine
class SmartDosingEngine:
    def __init__(self, drug_data):
        self.drug_data = drug_data
    
    def calculate_personalized_dose(self, drug_name, patient_info):
        """Calculate personalized dosage based on patient factors"""
        drug = self._find_drug(drug_name)
        if not drug:
            return None
        
        base_dose = self._parse_dosage(drug.get('adult_dosage', ''))
        adjustments = []
        final_dose = base_dose.copy() if base_dose else {}
        
        # Age adjustments
        age = patient_info.get('age', 30)
        if age < 18:
            pediatric_dose = drug.get('pediatric_dosage')
            if pediatric_dose and pediatric_dose != "Not recommended":
                adjustments.append(f"Pediatric dosing: {pediatric_dose}")
            else:
                adjustments.append("⚠️ Not recommended for pediatric use")
        elif age > 65:
            if final_dose.get('amount'):
                final_dose['amount'] *= 0.75  # Reduce dose for elderly
                adjustments.append("Reduced dose for elderly (25% reduction)")
        
        # Weight adjustments
        weight = patient_info.get('weight')
        if weight and weight < 50:
            adjustments.append("Consider dose reduction for low body weight")
        elif weight and weight > 100:
            adjustments.append("Consider dose adjustment for high body weight")
        
        # Kidney function
        if 'kidney_disease' in patient_info.get('conditions', []):
            adjustments.append("⚠️ Dose adjustment needed for kidney disease")
            if final_dose.get('amount'):
                final_dose['amount'] *= 0.5
        
        # Liver function
        if 'liver_disease' in patient_info.get('conditions', []):
            adjustments.append("⚠️ Dose adjustment needed for liver disease")
        
        # Pregnancy
        if patient_info.get('pregnancy', False):
            pregnancy_cat = drug.get('pregnancy_category', 'Unknown')
            if pregnancy_cat in ['D', 'X']:
                adjustments.append(f"🚫 Contraindicated in pregnancy (Category {pregnancy_cat})")
            elif pregnancy_cat in ['C']:
                adjustments.append(f"⚠️ Use with caution in pregnancy (Category {pregnancy_cat})")
        
        return {
            'original_dose': drug.get('adult_dosage'),
            'adjusted_dose': final_dose,
            'adjustments': adjustments,
            'max_daily_dose': drug.get('max_daily_dose'),
            'confidence': 0.8 if adjustments else 0.9
        }
    
    def _find_drug(self, drug_name):
        """Find drug in database"""
        for drug in self.drug_data['drugs']:
            if drug['name'].lower() == drug_name.lower():
                return drug
            if drug_name.lower() in [name.lower() for name in drug.get('generic_names', [])]:
                return drug
        return None
    
    def _parse_dosage(self, dosage_str):
        """Parse dosage string into structured format"""
        pattern = r'(\d+(?:-\d+)?)\s*(mg|g|ml|mcg)'
        match = re.search(pattern, dosage_str)
        if match:
            amount_str, unit = match.groups()
            if '-' in amount_str:
                amounts = [int(x) for x in amount_str.split('-')]
                amount = sum(amounts) / len(amounts)  # Average
            else:
                amount = int(amount_str)
            return {'amount': amount, 'unit': unit}
        return {}

# 🔥 FEATURE 5: Audio Input Processing
class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def process_audio_file(self, audio_file):
        """Process uploaded audio file to text"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.read())
                temp_path = tmp_file.name
            
            # Convert to wav if needed
            audio = AudioSegment.from_file(temp_path)
            audio = audio.set_channels(1).set_frame_rate(16000)  # Optimize for speech recognition
            wav_path = temp_path.replace('.wav', '_converted.wav')
            audio.export(wav_path, format="wav")
            
            # Transcribe
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
            
            # Cleanup
            os.unlink(temp_path)
            os.unlink(wav_path)
            
            return {"success": True, "text": text, "confidence": 0.8}
        
        except Exception as e:
            return {"success": False, "error": str(e), "text": ""}
    
    def record_microphone(self, duration=10):
        """Record from microphone (for future implementation)"""
        # This would require additional setup for web-based recording
        return {"success": False, "error": "Microphone recording not implemented in web version"}

# 🔥 FEATURE 6: Explainable AI Engine
class ExplainableAI:
    def __init__(self):
        self.explainer = LimeTextExplainer(class_names=['Safe', 'Risky'])
    
    def explain_decision(self, prescription_text, decision_factors):
        """Generate explanation for AI decisions"""
        explanations = []
        
        # Rule-based explanations
        for factor in decision_factors:
            if factor['type'] == 'interaction':
                explanations.append({
                    'reason': f"Drug interaction detected: {factor['drugs']}",
                    'mechanism': factor.get('mechanism', 'Unknown'),
                    'confidence': factor.get('confidence', 0.8),
                    'recommendation': factor.get('recommendation', 'Monitor closely')
                })
            elif factor['type'] == 'dosage':
                explanations.append({
                    'reason': f"Dosage concern: {factor['issue']}",
                    'recommendation': factor.get('recommendation', 'Verify dosage'),
                    'confidence': factor.get('confidence', 0.7)
                })
            elif factor['type'] == 'contraindication':
                explanations.append({
                    'reason': f"Contraindication: {factor['condition']}",
                    'recommendation': factor.get('recommendation', 'Consider alternative'),
                    'confidence': 0.9
                })
        
        return explanations
    
    def generate_report(self, explanations):
        """Generate detailed explanation report"""
        report = "## 🧠 AI Decision Explanation\n\n"
        
        for i, exp in enumerate(explanations, 1):
            confidence_emoji = "🔴" if exp['confidence'] > 0.8 else "🟡" if exp['confidence'] > 0.6 else "🟢"
            report += f"### {confidence_emoji} Factor {i}: {exp['reason']}\n"
            if 'mechanism' in exp:
                report += f"**Mechanism:** {exp['mechanism']}\n\n"
            report += f"**Recommendation:** {exp['recommendation']}\n"
            report += f"**Confidence:** {exp['confidence']:.1%}\n\n"
            report += "---\n\n"
        
        return report

# 🔥 FEATURE 7: Medication Schedule Planner
class MedicationScheduler:
    def generate_schedule(self, medications, patient_info):
        """Generate personalized medication schedule"""
        schedule = {}
        current_date = datetime.now()
        
        for med in medications:
            frequency = self._parse_frequency(med.get('frequency', 'once daily'))
            times_per_day = frequency['times_per_day']
            
            # Generate optimal timing
            if times_per_day == 1:
                times = ['08:00']
            elif times_per_day == 2:
                times = ['08:00', '20:00']
            elif times_per_day == 3:
                times = ['08:00', '13:00', '20:00']
            elif times_per_day == 4:
                times = ['08:00', '12:00', '16:00', '20:00']
            else:
                # Distribute evenly
                interval = 24 // times_per_day
                times = [f"{8 + i * interval:02d}:00" for i in range(times_per_day)]
            
            schedule[med['name']] = {
                'times': times,
                'dosage': med.get('dosage', 'As prescribed'),
                'with_food': self._requires_food(med['name']),
                'special_instructions': self._get_special_instructions(med['name'])
            }
        
        return schedule
    
    def _parse_frequency(self, freq_str):
        """Parse frequency string"""
        freq_str = freq_str.lower()
        if 'once' in freq_str or '1' in freq_str:
            return {'times_per_day': 1}
        elif 'twice' in freq_str or '2' in freq_str:
            return {'times_per_day': 2}
        elif 'thrice' in freq_str or 'three' in freq_str or '3' in freq_str:
            return {'times_per_day': 3}
        elif '4' in freq_str or 'four' in freq_str:
            return {'times_per_day': 4}
        else:
            return {'times_per_day': 2}  # Default
    
    def _requires_food(self, drug_name):
        """Check if drug should be taken with food"""
        with_food_drugs = ['ibuprofen', 'naproxen', 'aspirin', 'metformin']
        return drug_name.lower() in with_food_drugs
    
    def _get_special_instructions(self, drug_name):
        """Get special instructions for drug"""
        instructions = {
            'metformin': 'Take with meals to reduce stomach upset',
            'ibuprofen': 'Take with food or milk to prevent stomach irritation',
            'paracetamol': 'Can be taken with or without food'
        }
        return instructions.get(drug_name.lower(), 'Follow prescription instructions')

# Initialize components
@st.cache_resource
def initialize_ai_components():
    """Initialize all AI components"""
    drug_data = load_enhanced_drug_data()
    
    # Initialize components
    drug_graph = DrugInteractionGraph()
    drug_graph.build_graph(drug_data)
    
    ner_engine = PrescriptionNER()
    dosing_engine = SmartDosingEngine(drug_data)
    audio_processor = AudioProcessor()
    explainable_ai = ExplainableAI()
    scheduler = MedicationScheduler()
    
    return {
        'drug_data': drug_data,
        'drug_graph': drug_graph,
        'ner_engine': ner_engine,
        'dosing_engine': dosing_engine,
        'audio_processor': audio_processor,
        'explainable_ai': explainable_ai,
        'scheduler': scheduler
    }

# Load components
with st.spinner("🚀 Initializing Next-Gen AI Systems..."):
    components = initialize_ai_components()

st.success("✅ All AI systems loaded successfully!")

# 🔥 Enhanced OCR with better preprocessing
def advanced_ocr_processing(image):
    """Advanced OCR with multiple preprocessing techniques"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Multiple preprocessing approaches
        results = []
        
        # Method 1: Standard preprocessing
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh1 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        text1 = pytesseract.image_to_string(thresh1, config='--psm 6')
        results.append(text1)
        
        # Method 2: Morphological operations
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        text2 = pytesseract.image_to_string(morph, config='--psm 6')
        results.append(text2)
        
        # Method 3: Different PSM mode
        text3 = pytesseract.image_to_string(gray, config='--psm 3')
        results.append(text3)
        
        # Choose best result (longest valid text)
        best_text = max(results, key=lambda x: len(x.strip()) if x.strip() else 0)
        
        return best_text.strip() if best_text else "No text extracted"
        
    except Exception as e:
        return f"OCR Error: {str(e)}"

# 🔥 Main UI - Enhanced Sidebar
with st.sidebar:
    st.markdown("## 🎛️ System Controls")
    
    # AI Model Selection
    st.subheader("🤖 AI Engine")
    use_ai_model = st.checkbox("Enable Granite AI", value=True)
    model_choice = st.selectbox("Model:", ["granite", "small", "tiny"], index=0)
    
    # Input Methods
    st.subheader("📝 Input Methods")
    input_methods = st.multiselect(
        "Select input methods:",
        ["📝 Text", "📸 Image OCR", "🎤 Audio", "📁 File Upload"],
        default=["📝 Text", "📸 Image OCR"]
    )
    
    # Patient Information
    st.subheader("👤 Patient Profile")
    patient_age = st.slider("Age", 0, 120, 30)
    patient_weight = st.number_input("Weight (kg)", 0, 300, 70)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    is_pregnant = st.checkbox("Pregnant") if patient_gender == "Female" else False
    
    # Medical History
    st.subheader("📋 Medical History")
    medical_conditions = st.multiselect(
        "Medical Conditions:",
        ["Diabetes", "Hypertension", "Heart Disease", "Kidney Disease", 
         "Liver Disease", "Asthma", "Pregnancy", "Depression"]
    )
    
    allergies = st.text_area("Allergies:", placeholder="e.g., Penicillin, Shellfish")
    current_medications = st.text_area("Current Medications:", placeholder="List current meds...")
    
    # AI Features Toggle
    st.subheader("🔬 AI Features")
    enable_features = {
        "drug_interactions": st.checkbox("Drug Interaction Analysis", value=True),
        "smart_dosing": st.checkbox("Smart Dosing Engine", value=True),
        "schedule_planning": st.checkbox("Medication Scheduling", value=True),
        "explainable_ai": st.checkbox("Explainable AI", value=True),
        "cost_analysis": st.checkbox("Cost & Availability", value=True),
        "alternative_suggestions": st.checkbox("Alternative Suggestions", value=True)
    }

# 🔥 Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📝 Prescription Input")
    
    # Dynamic input based on selected methods
    prescription_text = ""
    
    if "📝 Text" in input_methods:
        with st.expander("📝 Text Input", expanded=True):
            prescription_text = st.text_area(
                "Enter prescription:",
                "Paracetamol 500mg twice daily for 3 days\nIbuprofen 400mg when needed for pain",
                height=100
            )
    
    if "📸 Image OCR" in input_methods:
        with st.expander("📸 Image OCR"):
            uploaded_image = st.file_uploader(
                "Upload prescription image:",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                col_img1, col_img2 = st.columns(2)
                
                with col_img1:
                    st.image(image, caption="Original", width=300)
                
                with col_img2:
                    with st.spinner("🔍 Advanced OCR Processing..."):
                        extracted_text = advanced_ocr_processing(image)
                        
                        if extracted_text:
                            st.success("✅ Text Extracted!")
                            prescription_text += "\n" + extracted_text
                            
                            st.text_area(
                                "Extracted Text (editable):",
                                value=extracted_text,
                                height=100,
                                key="ocr_result"
                            )
    
    if "🎤 Audio" in input_methods:
        with st.expander("🎤 Audio Input"):
            st.info("🎙️ Upload audio file containing prescription details")
            
            audio_file = st.file_uploader(
                "Choose audio file:",
                type=['wav', 'mp3', 'ogg', 'm4a']
            )
            
            if audio_file:
                with st.spinner("🎧 Processing Audio..."):
                    audio_result = components['audio_processor'].process_audio_file(audio_file)
                    
                    if audio_result['success']:
                        st.success("✅ Audio Transcribed!")
                        transcribed_text = st.text_area(
                            "Transcribed Text (editable):",
                            value=audio_result['text'],
                            height=100
                        )
                        prescription_text += "\n" + transcribed_text
                    else:
                        st.error(f"❌ Audio processing failed: {audio_result['error']}")
    
    if "📁 File Upload" in input_methods:
        with st.expander("📁 File Upload"):
            uploaded_file = st.file_uploader(
                "Upload prescription file:",
                type=['txt', 'pdf', 'doc', 'docx']
            )
            if uploaded_file:
                st.info("📄 File processing not fully implemented - please copy/paste content")

# 🔥 MAIN ANALYSIS BUTTON
if st.button("🚀 ANALYZE PRESCRIPTION", type="primary", use_container_width=True):
    if not prescription_text.strip():
        st.error("❌ Please provide prescription text through any input method")
    else:
        # Patient info compilation
        patient_info = {
            'age': patient_age,
            'weight': patient_weight,
            'gender': patient_gender,
            'pregnancy': is_pregnant,
            'conditions': medical_conditions,
            'allergies': allergies,
            'current_meds': current_medications.split('\n') if current_medications else []
        }
        
        with st.spinner("🧠 AI Brain Processing..."):
            # 🔥 FEATURE 1: NER Entity Extraction
            ner_results = components['ner_engine'].extract_entities(prescription_text)
            
            # 🔥 FEATURE 2: Extract drugs from text
            extracted_drugs = []
            for drug in components['drug_data']['drugs']:
                if drug['name'].lower() in prescription_text.lower():
                    # Extract dosage and frequency context
                    pattern = rf"{drug['name']}\s+(\d+\w+)?\s*(.*?)(?=\n|$|[A-Z][a-z]+\s+\d+)"
                    match = re.search(pattern, prescription_text, re.IGNORECASE)
                    context = match.group(0) if match else f"{drug['name']} (context not found)"
                    
                    extracted_drugs.append({
                        'name': drug['name'],
                        'context': context,
                        'raw_text': prescription_text
                    })
            
            # 🔥 Display Results in Multiple Tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🔍 Smart Analysis", 
                "⚡ Drug Interactions", 
                "💊 Smart Dosing", 
                "📅 Schedule Plan",
                "🧠 AI Explanations",
                "💰 Cost & Alternatives"
            ])
            
            with tab1:
                st.markdown("## 🔍 Comprehensive Smart Analysis")
                
                # Entity Recognition Results
                with st.expander("🎯 AI Entity Recognition", expanded=True):
                    if ner_results['ner_entities']:
                        df_entities = pd.DataFrame([
                            {
                                'Entity': ent['word'],
                                'Type': ent['entity_group'],
                                'Confidence': f"{ent['score']:.2%}",
                                'Start': ent['start'],
                                'End': ent['end']
                            }
                            for ent in ner_results['ner_entities']
                        ])
                        st.dataframe(df_entities, use_container_width=True)
                    else:
                        st.info("No named entities detected by NER model")
                
                # Structured Data Extraction
                with st.expander("📊 Structured Data Extraction"):
                    structured = ner_results['structured_data']
                    col_s1, col_s2, col_s3 = st.columns(3)
                    
                    with col_s1:
                        st.metric("Dosages Found", len(structured.get('dosage', [])))
                        if structured.get('dosage'):
                            for dose in structured['dosage']:
                                st.write(f"💊 {dose[0]} {dose[1]}")
                    
                    with col_s2:
                        st.metric("Frequencies Found", len(structured.get('frequency', [])))
                        if structured.get('frequency'):
                            for freq in structured['frequency']:
                                st.write(f"⏰ {freq[0]} {freq[1]}")
                    
                    with col_s3:
                        st.metric("Durations Found", len(structured.get('duration', [])))
                        if structured.get('duration'):
                            for dur in structured['duration']:
                                st.write(f"📅 {dur[0]} {dur[1]}")
                
                # Drug Safety Analysis
                st.markdown("### 🛡️ Safety Analysis")
                safety_issues = []
                
                for drug_info in extracted_drugs:
                    drug_name = drug_info['name']
                    drug_data = next((d for d in components['drug_data']['drugs'] if d['name'] == drug_name), None)
                    
                    if drug_data:
                        # Age-based checks
                        if patient_age < 18 and drug_data.get('pediatric_dosage') == "Not recommended":
                            safety_issues.append({
                                'type': 'age_contraindication',
                                'drug': drug_name,
                                'issue': 'Not recommended for pediatric use',
                                'severity': 'HIGH'
                            })
                        
                        # Pregnancy checks
                        if is_pregnant and drug_data.get('pregnancy_category') in ['D', 'X']:
                            safety_issues.append({
                                'type': 'pregnancy_risk',
                                'drug': drug_name,
                                'issue': f"Pregnancy Category {drug_data['pregnancy_category']}",
                                'severity': 'HIGH'
                            })
                        
                        # Condition-based contraindications
                        contraindications = drug_data.get('contraindications', '').lower()
                        for condition in medical_conditions:
                            if condition.lower() in contraindications:
                                safety_issues.append({
                                    'type': 'contraindication',
                                    'drug': drug_name,
                                    'issue': f"Contraindicated with {condition}",
                                    'severity': 'HIGH'
                                })
                
                # Display safety issues
                if safety_issues:
                    for issue in safety_issues:
                        severity_color = "🔴" if issue['severity'] == 'HIGH' else "🟡"
                        st.markdown(f"""
                        <div class="warning-card">
                            <h4>{severity_color} {issue['type'].replace('_', ' ').title()}</h4>
                            <p><strong>Drug:</strong> {issue['drug']}</p>
                            <p><strong>Issue:</strong> {issue['issue']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No major safety issues detected")
            
            with tab2:
                st.markdown("## ⚡ Drug Interaction Network Analysis")
                
                if enable_features['drug_interactions']:
                    # Get all drugs (prescribed + current)
                    all_drugs = [drug['name'] for drug in extracted_drugs]
                    if current_medications:
                        all_drugs.extend([med.strip() for med in current_medications.split('\n') if med.strip()])
                    
                    # Find interactions using graph
                    interactions = components['drug_graph'].find_interactions(all_drugs)
                    
                    if interactions:
                        st.error(f"⚠️ {len(interactions)} Drug Interaction(s) Detected!")
                        
                        # Create interaction network visualization
                        if len(all_drugs) > 1:
                            try:
                                fig = components['drug_graph'].visualize_interactions(all_drugs)
                                st.pyplot(fig)
                            except:
                                st.info("Network visualization unavailable")
                        
                        # Detailed interaction analysis
                        for interaction in interactions:
                            risk_color = {
                                'HIGH': '🔴',
                                'MODERATE': '🟡',
                                'LOW': '🟢'
                            }.get(interaction['risk_level'], '⚪')
                            
                            st.markdown(f"""
                            <div class="warning-card">
                                <h4>{risk_color} {interaction['risk_level']} Risk Interaction</h4>
                                <p><strong>Drugs:</strong> {interaction['drug1']} + {interaction['drug2']}</p>
                                <p><strong>Mechanism:</strong> {interaction['mechanism']}</p>
                                <p><strong>Risk Score:</strong> {interaction['weight']}/3</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("✅ No significant drug interactions detected")
                        
                        # Show interaction network anyway
                        if len(all_drugs) > 1:
                            try:
                                fig = components['drug_graph'].visualize_interactions(all_drugs)
                                st.pyplot(fig)
                                st.caption("Green nodes = Low risk, Orange = Moderate risk, Red = High risk")
                            except:
                                st.info("All drugs appear to be safe for combination")
                else:
                    st.info("Drug interaction analysis disabled")
            
            with tab3:
                st.markdown("## 💊 Smart Personalized Dosing")
                
                if enable_features['smart_dosing']:
                    for drug_info in extracted_drugs:
                        drug_name = drug_info['name']
                        
                        # Get personalized dosing recommendation
                        dosing_result = components['dosing_engine'].calculate_personalized_dose(
                            drug_name, patient_info
                        )
                        
                        if dosing_result:
                            st.markdown(f"### 💊 {drug_name}")
                            
                            col_d1, col_d2 = st.columns(2)
                            
                            with col_d1:
                                st.markdown("**📋 Standard Dosing:**")
                                st.info(f"Adult Dose: {dosing_result['original_dose']}")
                                if dosing_result['max_daily_dose']:
                                    st.warning(f"Max Daily: {dosing_result['max_daily_dose']}")
                            
                            with col_d2:
                                st.markdown("**🎯 Personalized Adjustments:**")
                                if dosing_result['adjustments']:
                                    for adj in dosing_result['adjustments']:
                                        if '⚠️' in adj or '🚫' in adj:
                                            st.error(adj)
                                        else:
                                            st.success(adj)
                                else:
                                    st.success("No adjustments needed")
                            
                            # Confidence meter
                            confidence = dosing_result['confidence']
                            st.progress(confidence, f"Confidence: {confidence:.1%}")
                            
                            st.markdown("---")
                else:
                    st.info("Smart dosing analysis disabled")
            
            with tab4:
                st.markdown("## 📅 Personalized Medication Schedule")
                
                if enable_features['schedule_planning']:
                    # Generate schedule
                    medications = []
                    for drug_info in extracted_drugs:
                        # Extract frequency from context or use default
                        frequency = "twice daily"  # Default
                        if "once" in drug_info['context'].lower():
                            frequency = "once daily"
                        elif "three times" in drug_info['context'].lower():
                            frequency = "three times daily"
                        
                        medications.append({
                            'name': drug_info['name'],
                            'frequency': frequency,
                            'dosage': '500mg'  # Extract from context if available
                        })
                    
                    if medications:
                        schedule = components['scheduler'].generate_schedule(medications, patient_info)
                        
                        # Display schedule in a nice format
                        st.markdown("### 🕐 Daily Medication Schedule")
                        
                        # Create a visual schedule
                        schedule_data = []
                        for med_name, med_schedule in schedule.items():
                            for time in med_schedule['times']:
                                schedule_data.append({
                                    'Time': time,
                                    'Medication': med_name,
                                    'Dosage': med_schedule['dosage'],
                                    'With Food': '🍽️' if med_schedule['with_food'] else '⭕',
                                    'Instructions': med_schedule['special_instructions']
                                })
                        
                        if schedule_data:
                            df_schedule = pd.DataFrame(schedule_data).sort_values('Time')
                            st.dataframe(df_schedule, use_container_width=True)
                            
                            # Calendar view
                            st.markdown("### 📅 Weekly Calendar View")
                            
                            # Create a simple calendar layout
                            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            cal_cols = st.columns(7)
                            
                            for i, day in enumerate(days):
                                with cal_cols[i]:
                                    st.markdown(f"**{day}**")
                                    for _, row in df_schedule.iterrows():
                                        st.write(f"{row['Time']}: {row['Medication']}")
                            
                            # Export options
                            st.markdown("### 📤 Export Schedule")
                            col_e1, col_e2, col_e3 = st.columns(3)
                            
                            with col_e1:
                                if st.button("📱 SMS Reminders"):
                                    st.info("SMS integration requires phone number setup")
                            
                            with col_e2:
                                if st.button("📅 Google Calendar"):
                                    st.info("Calendar export feature coming soon")
                            
                            with col_e3:
                                # CSV download
                                csv = df_schedule.to_csv(index=False)
                                st.download_button(
                                    label="💾 Download CSV",
                                    data=csv,
                                    file_name="medication_schedule.csv",
                                    mime="text/csv"
                                )
                    else:
                        st.info("No medications found to schedule")
                else:
                    st.info("Schedule planning disabled")
            
            with tab5:
                st.markdown("## 🧠 Explainable AI Analysis")
                
                if enable_features['explainable_ai']:
                    # Compile decision factors for explanation
                    decision_factors = []
                    
                    # Add interaction factors
                    if interactions:
                        for interaction in interactions:
                            decision_factors.append({
                                'type': 'interaction',
                                'drugs': f"{interaction['drug1']} + {interaction['drug2']}",
                                'mechanism': interaction['mechanism'],
                                'confidence': 0.9,
                                'recommendation': 'Monitor closely for adverse effects'
                            })
                    
                    # Add dosage factors
                    for drug_info in extracted_drugs:
                        if '500mg' in drug_info['context'] and 'ibuprofen' in drug_info['name'].lower():
                            decision_factors.append({
                                'type': 'dosage',
                                'issue': 'Standard dose for Ibuprofen',
                                'confidence': 0.8,
                                'recommendation': 'Dose appears appropriate'
                            })
                    
                    # Add contraindication factors
                    for issue in safety_issues:
                        if issue['type'] == 'contraindication':
                            decision_factors.append({
                                'type': 'contraindication',
                                'condition': issue['issue'],
                                'confidence': 0.95,
                                'recommendation': 'Consider alternative medication'
                            })
                    
                    # Generate explanations
                    explanations = components['explainable_ai'].explain_decision(
                        prescription_text, decision_factors
                    )
                    
                    if explanations:
                        # Display detailed explanations
                        explanation_report = components['explainable_ai'].generate_report(explanations)
                        st.markdown(explanation_report)
                        
                        # Confidence visualization
                        if len(explanations) > 1:
                            st.markdown("### 📊 Decision Confidence Analysis")
                            
                            confidence_data = pd.DataFrame([
                                {'Factor': f"Factor {i+1}", 'Confidence': exp['confidence']}
                                for i, exp in enumerate(explanations)
                            ])
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(confidence_data['Factor'], confidence_data['Confidence'])
                            
                            # Color bars based on confidence
                            for i, (bar, conf) in enumerate(zip(bars, confidence_data['Confidence'])):
                                if conf > 0.8:
                                    bar.set_color('red')
                                elif conf > 0.6:
                                    bar.set_color('orange')
                                else:
                                    bar.set_color('green')
                            
                            ax.set_ylabel('Confidence Level')
                            ax.set_title('AI Decision Confidence by Factor')
                            ax.set_ylim(0, 1)
                            
                            # Add percentage labels on bars
                            for bar, conf in zip(bars, confidence_data['Confidence']):
                                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                       f'{conf:.1%}', ha='center', va='bottom')
                            
                            st.pyplot(fig)
                    else:
                        st.info("No significant AI decisions to explain - prescription appears straightforward")
                else:
                    st.info("Explainable AI analysis disabled")
            
            with tab6:
                st.markdown("## 💰 Cost Analysis & Alternatives")
                
                if enable_features['cost_analysis'] or enable_features['alternative_suggestions']:
                    for drug_info in extracted_drugs:
                        drug_name = drug_info['name']
                        drug_data = next((d for d in components['drug_data']['drugs'] if d['name'] == drug_name), None)
                        
                        if drug_data:
                            st.markdown(f"### 💊 {drug_name}")
                            
                            col_c1, col_c2, col_c3 = st.columns(3)
                            
                            with col_c1:
                                st.markdown("**💰 Cost Information**")
                                cost_range = drug_data.get('cost_range', [0, 0])
                                st.metric("Price Range", f"${cost_range[0]} - ${cost_range[1]}")
                                
                                availability = drug_data.get('availability_score', 0.5)
                                availability_text = "High" if availability > 0.8 else "Medium" if availability > 0.5 else "Low"
                                st.metric("Availability", availability_text)
                            
                            with col_c2:
                                st.markdown("**🏪 Generic Options**")
                                if drug_data.get('generic_available'):
                                    st.success("✅ Generic Available")
                                    generic_savings = "30-80% savings"
                                    st.info(f"💡 {generic_savings}")
                                else:
                                    st.warning("❌ No Generic")
                            
                            with col_c3:
                                st.markdown("**🏷️ Brand Names**")
                                brands = drug_data.get('brand_names', [])
                                for brand in brands:
                                    st.write(f"• {brand}")
                            
                            # Alternative suggestions
                            if enable_features['alternative_suggestions']:
                                alternatives = components['drug_data']['drug_alternatives'].get(drug_name, [])
                                if alternatives:
                                    st.markdown("**🔄 Alternative Medications**")
                                    
                                    alt_data = []
                                    for alt in alternatives:
                                        alt_drug = next((d for d in components['drug_data']['drugs'] if d['name'] == alt), None)
                                        if alt_drug:
                                            alt_cost = alt_drug.get('cost_range', [0, 0])
                                            alt_data.append({
                                                'Alternative': alt,
                                                'Class': alt_drug.get('drug_class', 'Unknown'),
                                                'Cost Range': f"${alt_cost[0]}-${alt_cost[1]}",
                                                'Generic': '✅' if alt_drug.get('generic_available') else '❌'
                                            })
                                    
                                    if alt_data:
                                        df_alternatives = pd.DataFrame(alt_data)
                                        st.dataframe(df_alternatives, use_container_width=True)
                            
                            st.markdown("---")
                else:
                    st.info("Cost analysis and alternative suggestions disabled")

# 🔥 Right sidebar - Real-time insights
with col2:
    st.markdown("## 📊 Real-time Insights")
    
    # Quick stats
    if prescription_text:
        word_count = len(prescription_text.split())
        char_count = len(prescription_text)
        
        st.metric("Words", word_count)
        st.metric("Characters", char_count)
        
        # Drug detection preview
        detected_drugs = []
        for drug in components['drug_data']['drugs']:
            if drug['name'].lower() in prescription_text.lower():
                detected_drugs.append(drug['name'])
        
        st.metric("Drugs Detected", len(detected_drugs))
        
        if detected_drugs:
            st.markdown("**🔍 Detected Medications:**")
            for drug in detected_drugs:
                st.write(f"• {drug}")
    
    # System Status
    st.markdown("### 🔋 System Status")
    
    status_items = [
        ("🤖 AI Engine", "Active" if use_ai_model else "Standby"),
        ("📸 OCR System", "Ready"),
        ("🎤 Audio Processing", "Ready"),
        ("🧠 Knowledge Graph", "Loaded"),
        ("⚡ Drug Database", f"{len(components['drug_data']['drugs'])} drugs"),
    ]
    
    for item, status in status_items:
        st.write(f"{item}: **{status}**")
    
    # Recent Analysis History
    if st.session_state.analysis_history:
        st.markdown("### 📈 Recent Analysis")
        for i, analysis in enumerate(st.session_state.analysis_history[-3:], 1):
            st.write(f"{i}. {analysis[:50]}...")

# 🔥 Footer with advanced features
st.markdown("---")
st.markdown("## 🚀 Advanced Features")

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    <div class="feature-card">
        <h4>🧬 NER Engine</h4>
        <p>Named Entity Recognition for precise drug extraction</p>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class="feature-card">
        <h4>🕸️ Knowledge Graph</h4>
        <p>Network analysis of drug interactions</p>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class="feature-card">
        <h4>🎯 Smart Dosing</h4>
        <p>Personalized dosage recommendations</p>
    </div>
    """, unsafe_allow_html=True)

# Debug info
with st.expander("🔧 System Debug Info"):
    st.write("**Environment:**")
    st.write(f"- Platform: {platform.system()}")
    st.write(f"- Python packages loaded: ✅")
    st.write(f"- AI Components initialized: ✅")
    st.write(f"- Drug database: {len(components['drug_data']['drugs'])} entries")
    st.write(f"- Interaction rules: {len(components['drug_data'].get('high_risk_combinations', []))} combinations")

# Save analysis to history
if prescription_text and prescription_text not in [h for h in st.session_state.analysis_history]:
    st.session_state.analysis_history.append(prescription_text[:100])
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history.pop(0)