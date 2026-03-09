import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
import hashlib
import re
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title='Customer Churn Prediction',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)
sns.set_theme(style='whitegrid', palette='deep')

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    /* Reset Streamlit chrome */
    [data-testid="stToolbar"], footer, header, [data-testid="stDecoration"] { display: none !important; }
    .block-container { padding: 0 3.6rem 2.8rem 3.6rem; max-width: 100% !important; }
    body { background: transparent; }
    /* Background */
    .stApp {
        background: #060B1F;
        font-family: 'Poppins', sans-serif;
        color: #FFFFFF;
        min-height: 100vh;
    }
    .bg-layer {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        opacity: 0;
        transform: translate3d(0, 0, 0) scale(1.06);
        will-change: transform, opacity;
    }
    .auth-mode .bg-layer { opacity: 1; }
    .bg-base {
        background: linear-gradient(-45deg, #050914 0%, #0B1F6A 20%, #1D4ED8 40%, #3B82F6 55%, #F97316 78%, #FF7A1A 100%);
        background-size: 400% 400%;
        animation: baseDrift 20s ease-in-out infinite;
        z-index: 0;
    }
    .bg-base::before,
    .bg-base::after {
        content: "";
        position: absolute;
        inset: 0;
        opacity: 0.5;
        transform: translate3d(0, 0, 0) scale(1.08);
        animation: baseGlow 26s ease-in-out infinite;
    }
    .bg-base::before {
        background: radial-gradient(800px 600px at 20% 20%, rgba(56,189,248,0.18), transparent 60%);
    }
    .bg-base::after {
        background: radial-gradient(900px 700px at 80% 80%, rgba(249,115,22,0.16), transparent 60%);
        animation-delay: -8s;
    }
    .bg-grid {
        opacity: 0.1;
        background-image:
            linear-gradient(rgba(56,189,248,0.12) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56,189,248,0.12) 1px, transparent 1px);
        background-size: 60px 60px;
        transform: skewY(-8deg) scale(1.1);
        animation: gridDrift 30s linear infinite;
        z-index: 1;
    }
    .bg-particles {
        opacity: 0.12;
        background-image:
            radial-gradient(rgba(255,255,255,0.4) 1px, transparent 2px),
            radial-gradient(rgba(56,189,248,0.35) 1px, transparent 2px),
            radial-gradient(rgba(249,115,22,0.3) 1.5px, transparent 3px);
        background-size: 140px 140px, 220px 220px, 320px 320px;
        filter: blur(0.6px);
        animation: particleFloat 36s ease-in-out infinite;
        z-index: 2;
    }
    .bg-wave {
        opacity: 0.12;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1200' height='220' viewBox='0 0 1200 220'%3E%3Cpath d='M0 110 C200 40 400 180 600 110 C800 40 1000 180 1200 110' fill='none' stroke='rgba(56,189,248,0.55)' stroke-width='3'/%3E%3Cpath d='M0 160 C240 90 480 210 720 160 C960 90 1080 210 1200 160' fill='none' stroke='rgba(249,115,22,0.45)' stroke-width='2'/%3E%3C/svg%3E");
        background-repeat: repeat-x;
        background-size: 1200px 220px;
        animation: waveSlide 34s linear infinite;
        z-index: 3;
    }
    .bg-glow {
        opacity: 0.9;
        background:
            radial-gradient(520px at 8% 12%, rgba(56,189,248,0.45), transparent 60%),
            radial-gradient(520px at 92% 88%, rgba(249,115,22,0.4), transparent 60%);
        animation: glowPulse 16s ease-in-out infinite;
        z-index: 4;
    }
    .deploy-banner {
        position: fixed;
        top: 16px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        padding: 10px 16px;
        border-radius: 999px;
        background: rgba(10, 16, 36, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.3);
        backdrop-filter: blur(12px);
        box-shadow: 0 12px 28px rgba(5, 10, 26, 0.35), 0 0 18px rgba(59, 130, 246, 0.18);
        font-size: 14px;
        color: rgba(226, 232, 240, 0.9);
        letter-spacing: 0.01em;
        display: inline-flex;
        align-items: center;
        gap: 12px;
    }
    .deploy-banner .deploy-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 6px 14px;
        border-radius: 999px;
        background: linear-gradient(90deg, #F97316, #EC4899);
        color: #FFFFFF;
        font-weight: 600;
        text-decoration: none;
        box-shadow: 0 10px 18px rgba(249, 115, 22, 0.35);
        transition: transform 200ms ease, box-shadow 200ms ease, filter 200ms ease;
    }
    .deploy-banner .deploy-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 24px rgba(236, 72, 153, 0.35);
        filter: brightness(1.05);
    }
    @keyframes baseDrift {
        0% { transform: translate3d(-4%, 0%, 0) scale(1.06); }
        50% { transform: translate3d(4%, -3%, 0) scale(1.06); }
        100% { transform: translate3d(-4%, 0%, 0) scale(1.06); }
    }
    @keyframes baseGlow {
        0% { transform: translate3d(0, 0, 0) scale(1.08); opacity: 0.4; }
        50% { transform: translate3d(-2%, 1%, 0) scale(1.12); opacity: 0.6; }
        100% { transform: translate3d(0, 0, 0) scale(1.08); opacity: 0.4; }
    }
    @keyframes gridDrift {
        0% { transform: skewY(-8deg) translate3d(0, 0, 0) scale(1.1); }
        100% { transform: skewY(-8deg) translate3d(120px, -80px, 0) scale(1.1); }
    }
    @keyframes particleFloat {
        0% { transform: translate3d(0, 0, 0); }
        50% { transform: translate3d(-80px, 60px, 0); }
        100% { transform: translate3d(0, 0, 0); }
    }
    @keyframes waveSlide {
        0% { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(-400px, 0, 0); }
    }
    @keyframes glowPulse {
        0%, 100% { opacity: 0.65; }
        50% { opacity: 0.95; }
    }

    .ui-root { position: relative; z-index: 1; }
    .ui-hero h1 {
        background: linear-gradient(90deg, #FF2E2E, #E10600);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        text-shadow: 0 12px 28px rgba(5,10,26,0.35);
    }
    .ui-underline {
        width: 140px;
        height: 4px;
        border-radius: 999px;
        background: linear-gradient(90deg, #1E90FF, #FF6A00);
        margin-bottom: 16px;
        box-shadow: 0 0 16px rgba(30,144,255,0.55);
    }
    .ui-feature-list li::before {
        content: "🚀";
        width: 22px;
        height: 22px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(56, 189, 248, 0.2);
        color: #FFFFFF;
        font-size: 12px;
        box-shadow: 0 8px 16px rgba(56,189,248,0.35);
    }
    .ui-illustration {
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.14);
        border: none;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.16), 0 0 26px rgba(59, 130, 246, 0.2);
        padding: 12px;
        backdrop-filter: blur(18px);
        animation: chartFloatSlow 10s cubic-bezier(0.2, 0.8, 0.2, 1) infinite;
        transition: transform 250ms ease, box-shadow 250ms ease;
        position: relative;
        overflow: hidden;
    }
    .ui-illustration::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(600px 360px at 18% 18%, rgba(129, 140, 248, 0.24), transparent 65%),
                    radial-gradient(520px 320px at 82% 78%, rgba(96, 165, 250, 0.22), transparent 60%);
        opacity: 0.75;
        pointer-events: none;
    }
    .ui-illustration::after {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            radial-gradient(rgba(255,255,255,0.5) 1px, transparent 2px),
            radial-gradient(rgba(147,197,253,0.45) 1px, transparent 2px);
        background-size: 120px 120px, 180px 180px;
        opacity: 0.25;
        filter: blur(0.2px);
        animation: particleDrift 12s linear infinite;
        pointer-events: none;
    }
    .ui-illustration > * {
        position: relative;
        z-index: 1;
    }
    .ui-illustration:hover {
        transform: translate3d(0, -6px, 0);
        box-shadow: 0 22px 50px rgba(5, 10, 26, 0.32), 0 0 26px rgba(59, 130, 246, 0.22), inset 0 1px 2px rgba(255, 255, 255, 0.1);
    }
    .ui-illustration svg {
        transform: scale(1.1);
        transform-origin: center;
    }
    .ui-illustration svg .bar {
        transform-origin: center bottom;
        animation: barWave 4s cubic-bezier(0.2, 0.8, 0.2, 1) infinite,
                   barPulse 3.2s ease-in-out infinite;
        filter: drop-shadow(0 6px 12px rgba(59, 130, 246, 0.35));
    }
    .ui-illustration svg .bar-glow {
        filter: blur(12px);
        opacity: 0.55;
    }
    .ui-illustration svg .line-graph {
        stroke: rgba(224, 242, 254, 0.95);
        stroke-width: 3.1;
        filter: drop-shadow(0 0 12px rgba(147, 197, 253, 0.6));
    }
    .ui-illustration svg .line-dot {
        filter: drop-shadow(0 0 10px rgba(147, 197, 253, 0.7));
    }
    .ui-illustration svg .bar-sweep {
        opacity: 0.18;
        mix-blend-mode: screen;
        animation: sweepMove 4.5s cubic-bezier(0.2, 0.8, 0.2, 1) infinite;
    }
    @keyframes chartFloat {
        0%, 100% { transform: translate3d(0, 0, 0); }
        50% { transform: translate3d(0, -10px, 0); }
    }
    @keyframes chartFloatSlow {
        0%, 100% { transform: translate3d(0, 0, 0); }
        50% { transform: translate3d(0, -14px, 0); }
    }
    @keyframes barWave {
        0%, 100% { transform: scaleY(0.82); opacity: 0.85; }
        50% { transform: scaleY(1.08); opacity: 1; }
    }
    @keyframes barPulse {
        0%, 100% { filter: drop-shadow(0 8px 14px rgba(14, 116, 144, 0.35)); }
        50% { filter: drop-shadow(0 12px 18px rgba(96, 165, 250, 0.55)); }
    }
    @keyframes sweepMove {
        0% { transform: translateX(-35%); }
        100% { transform: translateX(35%); }
    }
    @keyframes particleDrift {
        0% { transform: translate3d(0, 0, 0); }
        50% { transform: translate3d(-20px, 16px, 0); }
        100% { transform: translate3d(0, 0, 0); }
    }
    .ui-card {
        background: rgba(10,20,50,0.65);
        border: 1px solid rgba(30,144,255,0.35);
        border-radius: 20px;
        padding: 26px;
        box-shadow: 0 22px 48px rgba(5,10,26,0.45), 0 0 18px rgba(30,144,255,0.25);
        backdrop-filter: blur(18px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .ui-card:hover {
        transform: translate3d(0, -2px, 0);
        box-shadow: 0 28px 58px rgba(5,10,26,0.55), 0 0 24px rgba(30,144,255,0.3);
    }
    .ui-card h3 { color: #FFFFFF; }
    .ui-card p { color: rgba(226,232,240,0.85); }
    .ui-card input[type="text"],
    .ui-card input[type="password"] {
        background: rgba(10,20,40,0.72) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(148,163,184,0.35) !important;
        color: #E2E8F0 !important;
        box-shadow: inset 0 0 0 1px rgba(8,18,38,0.45);
        caret-color: #FF6A00;
        transition: box-shadow 0.3s ease, transform 0.3s ease, border 0.3s ease;
    }
    .ui-card input[type="text"]:focus,
    .ui-card input[type="password"]:focus {
        border-color: rgba(30,144,255,0.8) !important;
        box-shadow: 0 0 0 3px rgba(30,144,255,0.25);
        transform: translate3d(0, -1px, 0);
    }
    .ui-card .stButton>button {
        background: linear-gradient(90deg, #1E90FF, #FF6A00) !important;
        border: none !important;
        box-shadow: 0 14px 26px rgba(30,144,255,0.35) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease, filter 0.3s ease;
    }
    .ui-card .stButton>button:hover {
        transform: translate3d(0, -2px, 0) !important;
        box-shadow: 0 20px 34px rgba(30,144,255,0.45) !important;
        filter: brightness(1.08);
    }
    /* Sidebar and metrics */
    [data-testid="stSidebar"] {
        background: rgba(6, 10, 24, 0.85);
        color: #E2E8F0;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 18px 0 36px rgba(5, 10, 26, 0.35);
        backdrop-filter: blur(16px);
    }
    [data-testid="stSidebar"] .block-container { padding: 2.2rem 1.4rem 2rem 1.4rem; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #E2E8F0;
        letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] h1 { font-size: 22px; margin-bottom: 0.5rem; }
    [data-testid="stSidebar"] h2 { font-size: 18px; margin-top: 1.4rem; }
    [data-testid="stSidebar"] h3 { font-size: 16px; margin-top: 1.2rem; }
    [data-testid="stSidebar"] .stMarkdown p { color: rgba(226,232,240,0.7); }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        background: rgba(15, 23, 42, 0.6);
        box-shadow: 0 10px 24px rgba(5, 10, 26, 0.22);
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] > div { color: #E2E8F0; }
    [data-testid="stSidebar"] .stButton>button {
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(90deg, #0EA5E9, #2563EB, #7C3AED);
        color: #FFFFFF;
        font-weight: 700;
        border: none;
        box-shadow: 0 12px 24px rgba(37,99,235,0.18);
        transition: transform 180ms ease, box-shadow 180ms ease;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 30px rgba(37,99,235,0.24);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: rgba(10, 16, 36, 0.8);
        border-radius: 16px;
        padding: 14px;
        border: 1px solid rgba(148, 163, 184, 0.28);
        color: #E2E8F0;
        box-shadow: 0 18px 30px rgba(5, 10, 26, 0.3);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        color: #E2E8F0;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        border-radius: 12px;
        background: rgba(255,255,255,0.08);
        color: #E2E8F0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    [data-testid="stSidebar"] hr { border-color: rgba(148, 163, 184, 0.35); }
    .stMetric {
        background: rgba(10, 16, 36, 0.65);
        padding: 0.75rem;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 18px 38px rgba(5, 10, 26, 0.35), 0 0 14px rgba(59, 130, 246, 0.12);
        color: #E2E8F0;
        backdrop-filter: blur(16px);
    }
    .stMetric [data-testid="stMetricLabel"] { color: rgba(226, 232, 240, 0.8); font-weight: 600; }
    .stMetric [data-testid="stMetricValue"] { color: #F8FAFC; font-weight: 700; }
    section.main h1,
    section.main h2,
    section.main h3 {
        color: #E2E8F0;
        text-shadow: 0 0 16px rgba(59, 130, 246, 0.25);
    }
    section.main p,
    section.main li,
    section.main label {
        color: rgba(226, 232, 240, 0.85);
    }
    section.main [data-testid="stDataFrame"],
    section.main [data-testid="stTable"],
    section.main [data-testid="stPyplot"],
    section.main [data-testid="stPlotlyChart"],
    section.main [data-testid="stAltairChart"] {
        background: rgba(10, 16, 36, 0.62);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 18px;
        padding: 12px 14px;
        box-shadow: 0 20px 42px rgba(5, 10, 26, 0.35), 0 0 16px rgba(59, 130, 246, 0.14);
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
        max-width: 920px;
        margin-left: auto;
        margin-right: auto;
    }
    section.main [data-testid="stPyplot"]::after,
    section.main [data-testid="stPlotlyChart"]::after,
    section.main [data-testid="stAltairChart"]::after {
        content: "";
        position: absolute;
        inset: 6px;
        border-radius: 14px;
        background: linear-gradient(120deg, rgba(56,189,248,0), rgba(56,189,248,0.25), rgba(99,102,241,0.18), rgba(236,72,153,0));
        mix-blend-mode: screen;
        opacity: 0.35;
        animation: chartSheen 10s ease-in-out infinite;
        pointer-events: none;
    }
    section.main [data-testid="stPyplot"] img {
        border-radius: 12px;
        filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.35));
        animation: chartPulse 7s ease-in-out infinite;
    }
    section.main [data-testid="stPlotlyChart"] svg .lines path,
    section.main [data-testid="stPlotlyChart"] svg .scatterlayer path {
        filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.45));
    }
    section.main [data-testid="stPlotlyChart"] svg .points path,
    section.main [data-testid="stPlotlyChart"] svg .points circle {
        animation: dataPointPulse 3.6s ease-in-out infinite;
        transform-origin: center;
    }
    section.main > div { padding-top: 12px; }

    /* =========================
       FAANG UI SYSTEM OVERRIDE
       ========================= */
    .bg-layer { display: none !important; }
    .stApp {
        background:
            radial-gradient(1200px 800px at 18% 12%, rgba(37, 64, 128, 0.28), transparent 65%),
            radial-gradient(1000px 700px at 82% 18%, rgba(15, 23, 42, 0.55), transparent 62%),
            radial-gradient(900px 700px at 70% 82%, rgba(76, 29, 149, 0.22), transparent 66%),
            linear-gradient(140deg, #050814 0%, #0b1430 40%, #0a0f24 70%, #050814 100%);
        background-size: 400% 400%;
        animation: deepSpaceShift 25s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    .stApp::before,
    .stApp::after {
        content: none;
    }
    .galaxy-bg {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .galaxy-layer {
        position: absolute;
        inset: -10% 0 0 -10%;
        width: 120%;
        height: 120%;
        will-change: transform;
        transform: translate3d(calc(var(--mx, 0) * 1px), calc(var(--my, 0) * 1px), 0);
    }
    .galaxy-stars-far {
        background-image: radial-gradient(rgba(255,255,255,0.22) 1px, transparent 2px);
        background-size: 200px 200px;
        opacity: 0.2;
        filter: blur(0.2px);
        animation: starFarDrift 40s linear infinite, starsTwinkle 9s ease-in-out infinite;
        transform: translate3d(calc(var(--mx, 0) * 2px), calc(var(--my, 0) * 2px), 0);
    }
    .galaxy-stars-mid {
        background-image:
            radial-gradient(rgba(255,255,255,0.5) 2px, transparent 3px),
            radial-gradient(rgba(255,255,255,0.35) 1px, transparent 2px);
        background-size: 160px 160px, 220px 220px;
        opacity: 0.3;
        filter: blur(0.3px) drop-shadow(0 0 6px rgba(255,255,255,0.35));
        animation: starMidDrift 30s linear infinite, starsTwinkle 7s ease-in-out infinite;
        transform: translate3d(calc(var(--mx, 0) * 4px), calc(var(--my, 0) * 4px), 0);
    }
    .galaxy-stars-near {
        background-image:
            radial-gradient(rgba(255,255,255,0.85) 2px, transparent 3px),
            radial-gradient(rgba(255,255,255,0.65) 3px, transparent 4px);
        background-size: 140px 140px, 200px 200px;
        opacity: 0.38;
        filter: blur(0.4px) drop-shadow(0 0 8px rgba(255,255,255,0.5));
        animation: starNearDrift 25s linear infinite, starsTwinkle 6s ease-in-out infinite;
        transform: translate3d(calc(var(--mx, 0) * 7px), calc(var(--my, 0) * 6px), 0);
    }
    .galaxy-nebula {
        background:
            radial-gradient(920px 720px at 18% 28%, rgba(56, 189, 248, 0.12), transparent 66%),
            radial-gradient(980px 740px at 78% 30%, rgba(124, 58, 237, 0.15), transparent 66%),
            radial-gradient(1040px 760px at 70% 78%, rgba(99, 102, 241, 0.12), transparent 68%),
            radial-gradient(820px 640px at 32% 70%, rgba(59, 130, 246, 0.1), transparent 66%),
            radial-gradient(760px 620px at 85% 65%, rgba(14, 165, 233, 0.1), transparent 66%),
            radial-gradient(720px 580px at 55% 20%, rgba(147, 51, 234, 0.12), transparent 64%);
        opacity: 0.45;
        filter: blur(2.4px);
        animation: nebulaFloat 35s ease-in-out infinite;
        transform: translate3d(calc(var(--mx, 0) * 3px), calc(var(--my, 0) * 3px), 0);
    }
    .galaxy-milkyway {
        background:
            radial-gradient(1200px 420px at 20% 55%, rgba(255, 255, 255, 0.15), transparent 70%),
            radial-gradient(1200px 420px at 80% 45%, rgba(191, 219, 254, 0.18), transparent 70%),
            linear-gradient(100deg, transparent 15%, rgba(255, 255, 255, 0.12) 40%, rgba(255, 255, 255, 0.2) 50%, rgba(147, 197, 253, 0.16) 60%, transparent 85%);
        opacity: 0.45;
        filter: blur(1.6px);
        mix-blend-mode: screen;
        animation: milkyWayDrift 40s ease-in-out infinite, milkyWayShimmer 9s ease-in-out infinite;
        transform: translate3d(calc(var(--mx, 0) * 4px), calc(var(--my, 0) * 4px), 0) rotate(-12deg) scale(1.05);
    }
    .galaxy-constellations {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1400' height='900' viewBox='0 0 1400 900'%3E%3Cdefs%3E%3ClinearGradient id='glow' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0%25' stop-color='rgba(147,197,253,0.55)'/%3E%3Cstop offset='100%25' stop-color='rgba(255,255,255,0.35)'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cg fill='none' stroke='url(%23glow)' stroke-width='1.1' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M120 180 L190 150 L250 210 L320 170 L380 230'/%3E%3Cpath d='M980 160 L1040 120 L1100 180 L1160 150 L1220 210'/%3E%3Cpath d='M200 620 L260 580 L320 640 L380 600 L460 660'/%3E%3Cpath d='M820 520 L880 480 L940 540 L1000 500 L1060 560'/%3E%3Cpath d='M560 320 L620 280 L700 320 L760 260 L840 300'/%3E%3C/g%3E%3Cg fill='rgba(255,255,255,0.7)'%3E%3Ccircle cx='120' cy='180' r='2'/%3E%3Ccircle cx='190' cy='150' r='2.4'/%3E%3Ccircle cx='250' cy='210' r='2.1'/%3E%3Ccircle cx='320' cy='170' r='2.3'/%3E%3Ccircle cx='380' cy='230' r='2'/%3E%3Ccircle cx='980' cy='160' r='2'/%3E%3Ccircle cx='1040' cy='120' r='2.5'/%3E%3Ccircle cx='1100' cy='180' r='2.2'/%3E%3Ccircle cx='1160' cy='150' r='2.3'/%3E%3Ccircle cx='1220' cy='210' r='2'/%3E%3Ccircle cx='200' cy='620' r='2'/%3E%3Ccircle cx='260' cy='580' r='2.4'/%3E%3Ccircle cx='320' cy='640' r='2.2'/%3E%3Ccircle cx='380' cy='600' r='2.4'/%3E%3Ccircle cx='460' cy='660' r='2'/%3E%3Ccircle cx='820' cy='520' r='2'/%3E%3Ccircle cx='880' cy='480' r='2.4'/%3E%3Ccircle cx='940' cy='540' r='2.2'/%3E%3Ccircle cx='1000' cy='500' r='2.4'/%3E%3Ccircle cx='1060' cy='560' r='2'/%3E%3Ccircle cx='560' cy='320' r='2'/%3E%3Ccircle cx='620' cy='280' r='2.4'/%3E%3Ccircle cx='700' cy='320' r='2.2'/%3E%3Ccircle cx='760' cy='260' r='2.4'/%3E%3Ccircle cx='840' cy='300' r='2'/%3E%3C/g%3E%3C/svg%3E");
        background-size: 1400px 900px;
        opacity: 0.32;
        mix-blend-mode: screen;
        filter: blur(0.35px) drop-shadow(0 0 6px rgba(147,197,253,0.3));
        animation: constellationsDrift 55s linear infinite, constellationPulse 8s ease-in-out infinite;
        transform: translate3d(calc(var(--mx, 0) * 2px), calc(var(--my, 0) * 2px), 0);
    }
    .galaxy-dust {
        background-image:
            radial-gradient(rgba(255,255,255,0.12) 1px, transparent 2px),
            radial-gradient(rgba(56,189,248,0.08) 1px, transparent 2px);
        background-size: 120px 120px, 180px 180px;
        opacity: 0.18;
        filter: blur(0.3px);
        animation: dustFlicker 5s ease-in-out infinite;
        transform: translate3d(calc(var(--mx, 0) * 2px), calc(var(--my, 0) * 2px), 0);
    }
    .galaxy-rockets {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1200' height='800' viewBox='0 0 1200 800'%3E%3Cg fill='none' stroke='rgba(255,255,255,0.22)' stroke-width='1.5'%3E%3Cpath d='M60 120 l20 -30 l20 30 -10 6 10 14 -20 -6 -20 6 10 -14z'/%3E%3Cpath d='M420 280 l18 -28 l18 28 -9 6 9 12 -18 -5 -18 5 9 -12z'/%3E%3Cpath d='M820 160 l20 -30 l20 30 -10 6 10 14 -20 -6 -20 6 10 -14z'/%3E%3Cpath d='M980 420 l16 -24 l16 24 -8 5 8 10 -16 -4 -16 4 8 -10z'/%3E%3Cpath d='M260 520 l14 -22 l14 22 -7 5 7 9 -14 -4 -14 4 7 -9z'/%3E%3C/g%3E%3C/svg%3E");
        background-size: 1000px 700px;
        opacity: 0.5;
        mix-blend-mode: screen;
        filter: blur(0.2px) drop-shadow(0 0 10px rgba(56,189,248,0.6));
        animation: rocketsDrift 45s linear infinite;
        transform: translate3d(calc(var(--mx, 0) * 3px), calc(var(--my, 0) * 3px), 0);
    }
    .galaxy-blackhole {
        background:
            radial-gradient(520px 520px at 50% 45%, rgba(11,17,32,0.55), transparent 60%),
            radial-gradient(420px 420px at 50% 45%, rgba(49,46,129,0.2), transparent 65%);
        opacity: 0.35;
        animation: blackholePulse 30s ease-in-out infinite;
        transform: translate3d(calc(var(--mx, 0) * 1px), calc(var(--my, 0) * 1px), 0);
    }
    .comet-layer {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
    }
    .comet {
        position: absolute;
        width: 240px;
        height: 3px;
        background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.85), rgba(56,189,248,0.45));
        filter: blur(0.2px) drop-shadow(0 0 8px rgba(56,189,248,0.5));
        opacity: 0;
        transform: translate3d(0, 0, 0) rotate(-25deg);
    }
    .comet.comet-1 { top: 12%; left: -30%; animation: cometFly 5s ease-in-out infinite; animation-delay: 1.5s; }
    .comet.comet-2 { top: 40%; left: -35%; animation: cometFly 5s ease-in-out infinite; animation-delay: 4.8s; }
    .comet.comet-3 { top: 70%; left: -40%; animation: cometFly 5s ease-in-out infinite; animation-delay: 8.2s; }
    .orb {
        position: fixed;
        width: 340px;
        height: 340px;
        border-radius: 50%;
        filter: blur(70px);
        opacity: 0.4;
        z-index: 0;
        animation: orbFloat 20s ease-in-out infinite;
        pointer-events: none;
    }
    .orb-1 { top: -60px; left: 8%; background: rgba(59,130,246,0.55); }
    .orb-2 { bottom: -80px; right: 10%; background: rgba(249,115,22,0.45); animation-delay: -6s; }
    .orb-3 { top: 30%; right: 40%; background: rgba(124,58,237,0.45); animation-delay: -12s; }

    /* Layer 1 - Reset */
    .ui-root, .ui-root * {
        box-shadow: none;
        text-shadow: none;
    }
    .ui-root *::before,
    .ui-root *::after {
        content: none;
    }

    /* Layer 2 - Design Tokens */
    .ui-root {
        --color-ink: #0F172A;
        --color-muted: rgba(15,23,42,0.7);
        --color-white: #FFFFFF;
        --color-primary: #2563EB;
        --color-accent: #9333EA;
        --color-warm: #F97316;
        --gradient-hero: linear-gradient(135deg, #0B1F6A 0%, #1D4ED8 32%, #7C3AED 68%, #F97316 100%);
        --radius-sm: 10px;
        --radius-md: 12px;
        --radius-lg: 24px;
        --shadow-soft: 0 18px 42px rgba(15,23,42,0.2);
        --shadow-strong: 0 26px 60px rgba(15,23,42,0.28);
        --space-1: 8px;
        --space-2: 12px;
        --space-3: 16px;
        --space-4: 20px;
        --space-5: 28px;
        --type-xl: 56px;
        --type-md: 18px;
        --ease-standard: cubic-bezier(0.2, 0.8, 0.2, 1);
        --duration-fast: 180ms;
        --duration-slow: 1200ms;
        color: var(--color-white);
        position: relative;
        z-index: 1;
    }

    /* Layer 3 - Layout System */
    .ui-root .ui-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
        gap: var(--space-5);
        align-items: center;
        min-height: 100vh;
    }
    @media (max-width: 980px) {
        .ui-root .ui-grid { grid-template-columns: 1fr; min-height: auto; padding-top: 6vh; }
    }

    /* Layer 4 - Component System */
    .ui-root .ui-hero {
        animation: uiFadeUp var(--duration-slow) var(--ease-standard) both;
    }
    .ui-root .ui-hero h1 {
        font-size: var(--type-xl);
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: var(--space-2);
        line-height: 1.04;
        text-shadow: 0 12px 28px rgba(5,10,26,0.35);
    }
    .ui-root .ui-hero p {
        font-size: var(--type-md);
        font-weight: 500;
        color: rgba(255,255,255,0.95);
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 10px 14px;
        border-radius: 14px;
        backdrop-filter: blur(6px);
        margin-bottom: var(--space-4);
    }
    .ui-root .ui-feature-list {
        list-style: none;
        padding: 0;
        margin: 0 0 var(--space-4) 0;
        display: grid;
        gap: var(--space-2);
    }
    .ui-root .ui-feature-list li {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        color: rgba(255,255,255,0.97);
        font-weight: 400;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.16);
        padding: 8px 12px;
        border-radius: 999px;
        backdrop-filter: blur(6px);
    }
    .ui-root .ui-feature-list li::before {
        content: "🚀";
        width: 22px;
        height: 22px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(56, 189, 248, 0.2);
        color: #FFFFFF;
        font-size: 12px;
        box-shadow: 0 8px 16px rgba(56,189,248,0.35);
    }
    .ui-root .ui-feature-list li::after {
        content: none;
    }
    .ui-root .ui-illustration {
        border-radius: 28px;
        background: rgba(10, 16, 36, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 30px 70px rgba(5, 10, 26, 0.45), 0 0 24px rgba(59, 130, 246, 0.25);
        padding: 22px;
        backdrop-filter: blur(18px);
        animation: chartFloat 11s ease-in-out infinite;
    }
    .ui-root .ui-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(255,255,255,0.55);
        border-radius: var(--radius-lg);
        padding: var(--space-4);
        box-shadow: var(--shadow-strong);
        backdrop-filter: blur(14px);
        animation: uiCardFloat 6s ease-in-out infinite;
    }
    .ui-root .ui-card h3 {
        color: var(--color-ink);
        font-size: 24px;
        font-weight: 700;
        margin-bottom: var(--space-1);
    }
    .ui-root .ui-card p {
        color: var(--color-muted);
        margin-bottom: var(--space-3);
    }
    .ui-root input[type="text"],
    .ui-root input[type="password"] {
        background: #F1F5F9 !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid rgba(148,163,184,0.6) !important;
        color: var(--color-ink) !important;
        padding-left: 44px !important;
        transition: box-shadow var(--duration-fast) var(--ease-standard), transform var(--duration-fast) var(--ease-standard);
    }
    .ui-root input[type="text"] { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' stroke='%232563EB' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2.5' y='4.5' width='15' height='11' rx='2.5'/%3E%3Cpath d='M3.5 7.5 10 11l6.5-3.5'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: 14px center; background-size: 18px; }
    .ui-root input[type="password"] { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' stroke='%232563EB' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='8' width='14' height='9' rx='2.5'/%3E%3Cpath d='M7 8V6a3 3 0 0 1 6 0v2'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: 14px center; background-size: 18px; }
    .ui-root input[type="text"]:focus,
    .ui-root input[type="password"]:focus {
        box-shadow: 0 0 0 3px rgba(59,130,246,0.25) !important;
        transform: translateY(-1px);
    }
    .ui-root .stButton>button {
        background: linear-gradient(90deg, #2563EB, #9333EA, #F97316) !important;
        background-size: 200% 200% !important;
        border-radius: 14px !important;
        box-shadow: 0 12px 28px rgba(30,58,138,0.2) !important;
        transition: transform var(--duration-fast) var(--ease-standard), box-shadow var(--duration-fast) var(--ease-standard);
        animation: buttonFlow 10s ease infinite;
    }
    .ui-root .stButton>button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 16px 32px rgba(30,58,138,0.26) !important;
    }
    .ui-root .ui-link {
        text-align: right;
        margin-top: -2px;
        margin-bottom: 12px;
    }
    .ui-root .ui-link a,
    .ui-root .ui-register a {
        color: #475569;
        text-decoration: none;
        position: relative;
    }
    .ui-root .ui-link a::after,
    .ui-root .ui-register a::after {
        content: "";
        position: absolute;
        left: 0;
        bottom: -2px;
        width: 0%;
        height: 2px;
        background: #2563EB;
        transition: width var(--duration-fast) var(--ease-standard);
    }
    .ui-root .ui-link a:hover::after,
    .ui-root .ui-register a:hover::after { width: 100%; }
    .ui-root .ui-register { text-align: center; color: #475569; font-weight: 500; }

    /* Layer 5 - Motion System */
    @keyframes uiFadeUp {
        from { opacity: 0; transform: translateY(18px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes uiFloat {
        0% { transform: translateY(0px); }
        50% { transform: translateY(18px); }
        100% { transform: translateY(0px); }
    }
    @keyframes uiCardFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes deepSpaceShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes dotsDrift {
        0% { background-position: 0 0, 12px 18px; }
        100% { background-position: 80px 120px, 60px 100px; }
    }
    @keyframes starsTwinkle {
        0%, 100% { opacity: 0.28; }
        50% { opacity: 0.6; }
    }
    @keyframes chartSheen {
        0% { transform: translate3d(-12%, -8%, 0); opacity: 0.2; }
        50% { transform: translate3d(12%, 8%, 0); opacity: 0.4; }
        100% { transform: translate3d(-12%, -8%, 0); opacity: 0.2; }
    }
    @keyframes chartPulse {
        0%, 100% { filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.28)); }
        50% { filter: drop-shadow(0 0 14px rgba(56, 189, 248, 0.5)); }
    }
    @keyframes dataPointPulse {
        0%, 100% { transform: scale(1); opacity: 0.85; }
        50% { transform: scale(1.12); opacity: 1; }
    }
    @keyframes starFarDrift {
        0% { background-position: 0 0; }
        100% { background-position: 0 -220px; }
    }
    @keyframes starMidDrift {
        0% { background-position: 0 0, 80px 60px; }
        100% { background-position: 0 -260px, 80px -200px; }
    }
    @keyframes starNearDrift {
        0% { background-position: 0 0, 120px 40px; }
        100% { background-position: 160px -240px, 240px -140px; }
    }
    @keyframes nebulaFloat {
        0% { transform: translate3d(0, 0, 0) scale(1); }
        50% { transform: translate3d(-2%, 1.5%, 0) scale(1.02); }
        100% { transform: translate3d(0, 0, 0) scale(1); }
    }
    @keyframes milkyWayDrift {
        0% { opacity: 0.35; transform: translate3d(-2%, 1%, 0) rotate(-12deg) scale(1.03); }
        50% { opacity: 0.55; transform: translate3d(2%, -1%, 0) rotate(-10deg) scale(1.06); }
        100% { opacity: 0.35; transform: translate3d(-2%, 1%, 0) rotate(-12deg) scale(1.03); }
    }
    @keyframes milkyWayShimmer {
        0%, 100% { filter: blur(1.8px) brightness(1); }
        50% { filter: blur(0.9px) brightness(1.35); }
    }
    @keyframes constellationsDrift {
        0% { background-position: 0 0; }
        100% { background-position: -240px 160px; }
    }
    @keyframes constellationPulse {
        0%, 100% { opacity: 0.32; }
        50% { opacity: 0.52; }
    }
    @keyframes blackholePulse {
        0%, 100% { opacity: 0.28; }
        50% { opacity: 0.4; }
    }
    @keyframes dustFlicker {
        0%, 100% { opacity: 0.12; }
        50% { opacity: 0.22; }
    }
    @keyframes rocketsDrift {
        0% { background-position: 0 0; }
        100% { background-position: -200px 160px; }
    }
    @keyframes cometFly {
        0% { opacity: 0; transform: translate3d(0, 0, 0) rotate(-25deg); }
        10% { opacity: 0.8; }
        50% { opacity: 0.9; }
        90% { opacity: 0.2; }
        100% { opacity: 0; transform: translate3d(140vw, 40vh, 0) rotate(-25deg); }
    }
    @keyframes glowPulse {
        0%, 100% { opacity: 0.75; }
        50% { opacity: 0.95; }
    }
    @keyframes orbFloat {
        0% { transform: translate3d(0, 0, 0); }
        50% { transform: translate3d(40px, -30px, 0); }
        100% { transform: translate3d(0, 0, 0); }
    }
    @keyframes buttonFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* =========================
       GALAXY LOGIN UI (AUTH MODE)
       ========================= */
    .auth-mode .stApp {
        background:
            radial-gradient(1200px 800px at 18% 12%, rgba(49, 46, 129, 0.3), transparent 65%),
            radial-gradient(1000px 700px at 82% 18%, rgba(30, 27, 75, 0.45), transparent 62%),
            linear-gradient(135deg, #0b1120 0%, #0f172a 35%, #1e1b4b 55%, #312e81 72%, #0b1120 100%);
        background-size: 400% 400%;
        animation: premiumShift 25s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    .auth-mode .stApp::before,
    .auth-mode .stApp::after {
        content: none;
    }
    .auth-mode .ui-root::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            radial-gradient(rgba(255,255,255,0.14) 1px, transparent 2px),
            radial-gradient(rgba(59,130,246,0.22) 1.5px, transparent 3px),
            linear-gradient(rgba(59,130,246,0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(59,130,246,0.08) 1px, transparent 1px);
        background-size: 200px 200px, 320px 320px, 64px 64px, 64px 64px;
        opacity: 0.22;
        animation: particleFloatSlow 36s ease-in-out infinite;
        z-index: 0;
    }

    .auth-mode .ui-hero h1 {
        font-size: 60px;
        font-weight: 700;
        letter-spacing: -0.02em;
        text-align: center;
        background: linear-gradient(90deg, #FF2E2E, #E10600);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        text-shadow: 0 10px 24px rgba(8, 12, 28, 0.5);
    }
    .auth-mode .ui-underline {
        background: linear-gradient(90deg, #3B82F6, #FB923C);
        box-shadow: 0 0 18px rgba(59,130,246,0.6);
    }
    .auth-mode .ui-feature-list li::before {
        background: #2563EB;
        box-shadow: 0 0 16px rgba(59,130,246,0.8), 0 0 26px rgba(251,146,60,0.45);
    }
    .auth-mode .ui-illustration {
        background: rgba(10, 18, 40, 0.2);
        border: none;
        border-radius: 30px;
        box-shadow: 0 22px 52px rgba(6, 10, 28, 0.35), 0 0 26px rgba(99, 102, 241, 0.22);
        position: relative;
        backdrop-filter: blur(18px);
        animation: chartFloatSlow 10s cubic-bezier(0.2, 0.8, 0.2, 1) infinite;
        overflow: hidden;
        padding: 12px;
    }
    .auth-mode .ui-illustration::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(600px 360px at 18% 18%, rgba(129, 140, 248, 0.22), transparent 65%),
                    radial-gradient(520px 320px at 82% 78%, rgba(96, 165, 250, 0.2), transparent 60%);
        opacity: 0.75;
        pointer-events: none;
    }
    .auth-mode .ui-illustration > * {
        position: relative;
        z-index: 1;
    }
    .auth-mode .ui-illustration::after {
        content: "";
        position: absolute;
        left: 8%;
        right: 8%;
        bottom: 10px;
        height: 6px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(251,146,60,0), rgba(251,146,60,0.6), rgba(251,146,60,0));
        opacity: 0.8;
        filter: blur(6px);
    }
    .auth-mode .ui-illustration svg {
        filter: drop-shadow(0 0 12px rgba(59,130,246,0.4));
    }
    .auth-mode .ui-illustration svg .bar {
        filter: drop-shadow(0 0 12px rgba(251,146,60,0.3));
    }
    .auth-mode .ui-illustration svg .line-graph {
        stroke-linecap: round;
        stroke-linejoin: round;
        animation: lineGrow 2.2s ease-in-out infinite;
    }
    .auth-mode .ui-illustration svg circle {
        animation: nodePulse 2.8s ease-in-out infinite;
        filter: drop-shadow(0 0 10px rgba(59,130,246,0.5));
    }

    .auth-mode .ui-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(59,130,246,0.35);
        border-radius: 22px;
        box-shadow: 0 36px 90px rgba(6, 10, 28, 0.75), 0 0 28px rgba(59,130,246,0.3);
        backdrop-filter: blur(22px);
        background-image: none !important;
    }
    .auth-mode .ui-card .ui-card-header {
        display: contents;
        position: static;
        margin: 0;
        padding: 0;
        background: transparent !important;
        background-image: none !important;
        border-radius: 0;
        border-bottom: none;
        backdrop-filter: none;
    }
    .auth-mode .ui-card .ui-card-header::after {
        content: none;
    }
    .auth-mode .ui-card .ui-card-header h3 {
        margin: 0 !important;
        padding: 0 !important;
        color: #E6F0FF;
        text-align: center;
        font-weight: 700;
        letter-spacing: 0.01em;
        text-shadow: 0 6px 18px rgba(59,130,246,0.35);
    }
    .auth-mode .ui-card .ui-card-body {
        margin: 0;
        padding: 20px 22px 22px 22px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 0 0 22px 22px;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
    }
    .auth-mode .ui-card .ui-card-body p {
        margin: 0 0 14px 0 !important;
        color: rgba(226, 232, 240, 0.8);
    }
    .auth-mode .ui-card .ui-card-body-actions {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 10px;
        color: rgba(226, 232, 240, 0.7);
        font-size: 14px;
    }
    .auth-mode .ui-card .ui-card-body-actions .stButton>button {
        padding: 0 !important;
        font-weight: 600;
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        color: #93C5FD !important;
    }
    .auth-mode .ui-card .ui-card-body-actions .stButton>button:hover {
        color: #BFDBFE !important;
        transform: none !important;
    }
    .auth-mode .ui-card::before,
    .auth-mode .ui-card::after {
        content: none !important;
        display: none !important;
        background: none !important;
        box-shadow: none !important;
        height: 0 !important;
    }
    .auth-mode .stForm::before,
    .auth-mode .stForm::after,
    .auth-mode [data-testid="stForm"]::before,
    .auth-mode [data-testid="stForm"]::after,
    .auth-mode [data-testid="stContainer"]::before,
    .auth-mode [data-testid="stContainer"]::after,
    .auth-mode [data-testid="stVerticalBlock"]::before,
    .auth-mode [data-testid="stVerticalBlock"]::after {
        content: none !important;
        display: none !important;
        background: none !important;
        box-shadow: none !important;
        height: 0 !important;
    }
    .auth-mode .ui-card [data-baseweb="tab-list"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
    }
    .auth-mode .ui-card [data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
    }
    .auth-mode .ui-card [data-baseweb="tab-border"],
    .auth-mode .ui-card [data-baseweb="tab-highlight"],
    .auth-mode .ui-card [role="tablist"]::before,
    .auth-mode .ui-card [role="tablist"]::after {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    .auth-mode .ui-card [role="tablist"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .auth-mode .ui-card [data-testid="stTabs"],
    .auth-mode .ui-card [data-testid="stTabs"] > div,
    .auth-mode .ui-card [data-testid="stTabs"] > div > div {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    .auth-mode .ui-card [data-testid="stTabs"] > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .auth-mode .ui-card [data-testid="stTabs"] [data-baseweb="tab-list"],
    .auth-mode .ui-card [data-testid="stTabs"] [data-baseweb="tab-list"] > div,
    .auth-mode .ui-card [data-testid="stTabs"] [data-baseweb="tab-list"]::before,
    .auth-mode .ui-card [data-testid="stTabs"] [data-baseweb="tab-list"]::after {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        border-radius: 0 !important;
        height: auto !important;
        min-height: 0 !important;
    }
    .auth-mode [data-testid="stTabs"],
    .auth-mode [data-testid="stTabs"] > div,
    .auth-mode [data-testid="stTabs"] > div > div {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    .auth-mode [data-testid="stTabs"] [data-baseweb="tab-list"],
    .auth-mode [data-testid="stTabs"] [role="tablist"],
    .auth-mode [data-testid="stTabs"] [data-baseweb="tab-border"],
    .auth-mode [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    .auth-mode [data-testid="stTabs"] [role="tablist"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    .auth-mode [data-testid="stRadio"] > div,
    .auth-mode [data-testid="stRadio"] div[role="radiogroup"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .auth-mode [data-testid="stRadio"] label,
    .auth-mode [data-testid="stRadio"] [data-baseweb="radio"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    .auth-mode [data-testid="baseButton-secondary"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #93C5FD !important;
        padding: 0 !important;
        font-weight: 600;
    }
    .auth-mode [data-testid="baseButton-secondary"]:hover {
        color: #BFDBFE !important;
        transform: none !important;
    }
    .auth-mode .ui-card [data-testid="stRadio"],
    .auth-mode .ui-card [role="radiogroup"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .auth-mode .ui-card h3 { color: #E6F0FF; }
    .auth-mode .ui-card p { color: rgba(226, 232, 240, 0.85); }
    .auth-mode .ui-card:empty {
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
        background: none !important;
    }
    .auth-mode .ui-card [data-testid="stElementContainer"]:first-child {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .auth-mode .ui-card [data-testid="stElementContainer"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .auth-mode .ui-card [data-testid="stMarkdownContainer"]::before,
    .auth-mode .ui-card [data-testid="stMarkdownContainer"]::after,
    .auth-mode .ui-card [data-testid="stContainer"]::before,
    .auth-mode .ui-card [data-testid="stContainer"]::after,
    .auth-mode .ui-card [data-testid="stVerticalBlock"]::before,
    .auth-mode .ui-card [data-testid="stVerticalBlock"]::after,
    .auth-mode .ui-card [data-testid="stForm"]::before,
    .auth-mode .ui-card [data-testid="stForm"]::after,
    .auth-mode .ui-card .stForm::before,
    .auth-mode .ui-card .stForm::after {
        content: none !important;
        display: none !important;
        background: none !important;
        box-shadow: none !important;
        height: 0 !important;
    }

    .auth-mode .ui-root input[type="text"],
    .auth-mode .ui-root input[type="password"] {
        background: rgba(9, 15, 34, 0.9) !important;
        border: 1px solid rgba(59,130,246,0.35) !important;
        color: #E2E8F0 !important;
        box-shadow: inset 0 0 0 1px rgba(8,18,38,0.55), 0 0 12px rgba(37,99,235,0.12);
    }
    .auth-mode .ui-root input[type="text"]:focus,
    .auth-mode .ui-root input[type="password"]:focus {
        border-color: rgba(59,130,246,0.85) !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.35) !important;
        transform: translateY(-1px);
    }
    .auth-mode .ui-root .stButton>button {
        background: linear-gradient(90deg, #F97316, #EC4899) !important;
        box-shadow: 0 18px 34px rgba(249,115,22,0.35), 0 0 22px rgba(236,72,153,0.3) !important;
    }
    .auth-mode .ui-root .stButton>button:hover {
        transform: translateY(-2px) scale(1.01);
        filter: brightness(1.08);
    }

    /* DOM-safe global button color shift */
    [data-testid="stButton"] button,
    [data-testid="baseButton-primary"] button,
    [data-testid="baseButton-secondary"] button,
    [data-testid="baseButton-primary"] [role="button"],
    [data-testid="baseButton-secondary"] [role="button"] {
        background: linear-gradient(90deg, #38BDF8, #22D3EE, #A855F7, #EC4899) !important;
        background-size: 300% 300% !important;
        color: #FFFFFF !important;
        animation: buttonFlow 6s cubic-bezier(0.2, 0.8, 0.2, 1) infinite !important;
        transition: transform 200ms ease, box-shadow 200ms ease, filter 200ms ease, background-position 600ms ease;
    }
    [data-testid="stButton"] button:hover,
    [data-testid="baseButton-primary"] button:hover,
    [data-testid="baseButton-secondary"] button:hover,
    [data-testid="baseButton-primary"] [role="button"]:hover,
    [data-testid="baseButton-secondary"] [role="button"]:hover {
        background-position: 100% 50% !important;
        filter: brightness(1.08) hue-rotate(12deg);
    }

    @keyframes starDrift {
        0% { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(0, 140px, 0); }
    }
    @keyframes nebulaPulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 0.8; }
    }
    @keyframes particleFloatSlow {
        0% { transform: translate3d(0, 0, 0); }
        50% { transform: translate3d(-60px, 40px, 0); }
        100% { transform: translate3d(0, 0, 0); }
    }
    @keyframes streakSlide {
        0% { transform: translate3d(0, 0, 0); opacity: 0.2; }
        50% { transform: translate3d(6%, 0, 0); opacity: 0.4; }
        100% { transform: translate3d(0, 0, 0); opacity: 0.2; }
    }
    @keyframes gradientFloat {
        0% { transform: translate3d(0, 0, 0); }
        50% { transform: translate3d(-3%, 2%, 0); }
        100% { transform: translate3d(0, 0, 0); }
    }
    @keyframes heroFloat {
        0%, 100% { transform: translate3d(0, 0, 0); }
        50% { transform: translate3d(0, -8px, 0); }
    }
    @keyframes premiumShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes lineGrow {
        0%, 100% { stroke-dasharray: 0 400; opacity: 0.8; }
        50% { stroke-dasharray: 400 0; opacity: 1; }
    }
    @keyframes nodePulse {
        0%, 100% { transform: scale(1); opacity: 0.85; }
        50% { transform: scale(1.12); opacity: 1; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="deploy-banner"><span>Deploy this app</span>'
    '<a class="deploy-button" href="https://share.streamlit.io" target="_blank" rel="noopener noreferrer">Deploy</a></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="galaxy-bg">'
    '<div class="galaxy-layer galaxy-nebula"></div>'
    '<div class="galaxy-layer galaxy-milkyway"></div>'
    '<div class="galaxy-layer galaxy-constellations"></div>'
    '<div class="galaxy-layer galaxy-blackhole"></div>'
    '<div class="galaxy-layer galaxy-rockets"></div>'
    '<div class="galaxy-layer galaxy-stars-far"></div>'
    '<div class="galaxy-layer galaxy-stars-mid"></div>'
    '<div class="galaxy-layer galaxy-stars-near"></div>'
    '<div class="galaxy-layer galaxy-dust"></div>'
    '<div class="comet-layer">'
    '<div class="comet comet-1"></div>'
    '<div class="comet comet-2"></div>'
    '<div class="comet comet-3"></div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <script>
    (function () {
        const root = document.documentElement;
        let tx = 0, ty = 0, cx = 0, cy = 0;
        const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
        window.addEventListener('mousemove', (e) => {
            const x = e.clientX / window.innerWidth - 0.5;
            const y = e.clientY / window.innerHeight - 0.5;
            tx = clamp(x * 18, -12, 12);
            ty = clamp(y * 18, -12, 12);
        }, { passive: true });
        const tick = () => {
            cx += (tx - cx) * 0.08;
            cy += (ty - cy) * 0.08;
            root.style.setProperty('--mx', cx.toFixed(2));
            root.style.setProperty('--my', cy.toFixed(2));
            requestAnimationFrame(tick);
        };
        tick();
    })();
    </script>
    """,
    unsafe_allow_html=True,
)


def get_db_connection():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, 'users.db')
    return sqlite3.connect(db_path, check_same_thread=False)


def init_user_table():
    conn = get_db_connection()
    cur = conn.cursor()
    # Create a base table if missing, then migrate columns as needed
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    # Migration: ensure required columns exist for older DBs
    cur.execute('PRAGMA table_info(users)')
    cols = [row[1] for row in cur.fetchall()]
    if 'password_hash' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
    if 'full_name' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN full_name TEXT')
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def create_user(username: str, full_name: str, password: str):
    conn = get_db_connection()
    cur = conn.cursor()
    # Detect current columns to avoid schema mismatch
    cur.execute('PRAGMA table_info(users)')
    cols = [row[1] for row in cur.fetchall()]
    # Ensure required columns exist
    if 'password_hash' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
        cur.execute('PRAGMA table_info(users)')
        cols = [row[1] for row in cur.fetchall()]
    if 'full_name' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN full_name TEXT')
        cur.execute('PRAGMA table_info(users)')
        cols = [row[1] for row in cur.fetchall()]
    conn.commit()
    try:
        cur.execute(
            'INSERT INTO users (username, full_name, password_hash) VALUES (?, ?, ?)',
            (username.strip().lower(), full_name.strip(), hash_password(password)),
        )
        conn.commit()
        return True, 'Registration successful. You can now log in.'
    except sqlite3.IntegrityError:
        return False, 'Username already exists. Please choose a different one.'
    finally:
        conn.close()


def authenticate_user(username: str, password: str):
    conn = get_db_connection()
    cur = conn.cursor()
    # Ensure columns exist for older DBs
    cur.execute('PRAGMA table_info(users)')
    cols = [row[1] for row in cur.fetchall()]
    if 'password_hash' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
    if 'full_name' not in cols:
        cur.execute('ALTER TABLE users ADD COLUMN full_name TEXT')
    conn.commit()
    cur.execute(
        'SELECT id, username, full_name FROM users WHERE username = ? AND password_hash = ?',
        (username.strip().lower(), hash_password(password)),
    )
    row = cur.fetchone()
    conn.close()
    return row


def valid_username(username: str) -> bool:
    return bool(re.fullmatch(r'[A-Za-z0-9_.-]{4,30}', username or ''))


def valid_password(password: str) -> bool:
    if password is None:
        return False
    has_min_len = len(password) >= 8
    has_alpha = bool(re.search(r'[A-Za-z]', password))
    has_digit = bool(re.search(r'\d', password))
    return has_min_len and has_alpha and has_digit


init_user_table()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'full_name' not in st.session_state:
    st.session_state.full_name = ''
if 'auth_mode' not in st.session_state:
    st.session_state.auth_mode = 'Login'

if not st.session_state.logged_in:
    st.markdown('<script>document.body.classList.add("auth-mode");</script>', unsafe_allow_html=True)
    st.markdown(
        '<div class="orb orb-1"></div><div class="orb orb-2"></div><div class="orb orb-3"></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ui-root">', unsafe_allow_html=True)
    st.markdown('<div class="ui-grid">', unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1], gap='large')

    with col1:
        st.markdown(
            """
            <div class="ui-hero">
                <h1>CUSTOMER CHURN PREDICTION SYSTEM</h1>
                <div class="ui-underline"></div>
                <p>A premium AI platform to predict churn, reveal segments, and turn insight into action.</p>
                <ul class="ui-feature-list">
                    <li>Churn forecasting with real-time clarity</li>
                    <li>Segmentation that aligns teams and outcomes</li>
                    <li>Explainable intelligence leaders can trust</li>
                </ul>
                <div class="ui-illustration">
                    <svg class="hero-chart" viewBox="0 0 720 320" width="100%" height="460" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
                        <defs>
                            <linearGradient id="glassBg" x1="0" y1="0" x2="1" y2="1">
                                <stop offset="0%" stop-color="rgba(12,20,42,0.85)" />
                                <stop offset="100%" stop-color="rgba(10,20,40,0.55)" />
                            </linearGradient>
                            <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stop-color="#60A5FA" />
                                <stop offset="55%" stop-color="#22D3EE" />
                                <stop offset="100%" stop-color="#FDBA74" />
                            </linearGradient>
                            <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stop-color="rgba(59,130,246,0.35)" />
                                <stop offset="100%" stop-color="rgba(59,130,246,0)" />
                            </linearGradient>
                            <linearGradient id="sweepGradient" x1="0" y1="0" x2="1" y2="0">
                                <stop offset="0%" stop-color="rgba(255,255,255,0)" />
                                <stop offset="50%" stop-color="rgba(255,255,255,0.5)" />
                                <stop offset="100%" stop-color="rgba(255,255,255,0)" />
                            </linearGradient>
                            <radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stop-color="#93C5FD" />
                                <stop offset="100%" stop-color="rgba(59,130,246,0)" />
                            </radialGradient>
                            <clipPath id="chartClip">
                                <rect x="34" y="56" width="652" height="210" rx="20" />
                            </clipPath>
                        </defs>
                        <rect x="10" y="12" width="700" height="296" rx="30" fill="url(#glassBg)" stroke="rgba(255,255,255,0.12)" />
                        <rect x="22" y="24" width="676" height="272" rx="26" fill="rgba(10,20,40,0.42)" stroke="rgba(255,255,255,0.08)" />
                        <g opacity="0.25">
                            <circle cx="90" cy="72" r="1" fill="#FFFFFF" />
                            <circle cx="160" cy="120" r="1" fill="#FFFFFF" />
                            <circle cx="220" cy="80" r="1" fill="#FFFFFF" />
                            <circle cx="560" cy="90" r="1" fill="#FFFFFF" />
                            <circle cx="620" cy="140" r="1" fill="#FFFFFF" />
                            <circle cx="460" cy="60" r="1" fill="#FFFFFF" />
                        </g>
                        <g opacity="0.22">
                            <path d="M60 96 H660" stroke="rgba(148,163,184,0.25)" stroke-width="1" />
                            <path d="M60 136 H660" stroke="rgba(148,163,184,0.18)" stroke-width="1" />
                            <path d="M60 176 H660" stroke="rgba(148,163,184,0.18)" stroke-width="1" />
                            <path d="M60 216 H660" stroke="rgba(148,163,184,0.12)" stroke-width="1" />
                        </g>
                        <g clip-path="url(#chartClip)">
                            <path d="M70 238 L140 204 L220 214 L300 170 L380 156 L460 190 L540 142 L640 164 L640 266 L70 266 Z" fill="url(#areaFill)" />
                            <path class="line-graph" d="M70 238 L140 204 L220 214 L300 170 L380 156 L460 190 L540 142 L640 164" fill="none" stroke-linecap="round" stroke-linejoin="round" />
                            <circle class="line-dot" r="4" fill="#E0F2FE">
                                <animateMotion dur="4s" repeatCount="indefinite" path="M70 238 L140 204 L220 214 L300 170 L380 156 L460 190 L540 142 L640 164" />
                            </circle>
                            <circle cx="540" cy="142" r="10" fill="rgba(56,189,248,0.2)" />
                            <circle cx="540" cy="142" r="4" fill="#93C5FD" />
                            <rect class="bar bar-glow" x="88" y="182" width="38" height="86" rx="12" fill="url(#barGradient)" style="animation-delay: 0.1s" />
                            <rect class="bar" x="92" y="186" width="30" height="78" rx="12" fill="url(#barGradient)" style="animation-delay: 0.1s" />
                            <rect class="bar bar-glow" x="150" y="164" width="38" height="104" rx="12" fill="url(#barGradient)" style="animation-delay: 0.6s" />
                            <rect class="bar" x="154" y="168" width="30" height="96" rx="12" fill="url(#barGradient)" style="animation-delay: 0.6s" />
                            <rect class="bar bar-glow" x="212" y="190" width="38" height="78" rx="12" fill="url(#barGradient)" style="animation-delay: 1.1s" />
                            <rect class="bar" x="216" y="194" width="30" height="70" rx="12" fill="url(#barGradient)" style="animation-delay: 1.1s" />
                            <rect class="bar bar-glow" x="274" y="142" width="38" height="126" rx="12" fill="url(#barGradient)" style="animation-delay: 1.6s" />
                            <rect class="bar" x="278" y="146" width="30" height="118" rx="12" fill="url(#barGradient)" style="animation-delay: 1.6s" />
                            <rect class="bar bar-glow" x="336" y="154" width="38" height="114" rx="12" fill="url(#barGradient)" style="animation-delay: 2.1s" />
                            <rect class="bar" x="340" y="158" width="30" height="106" rx="12" fill="url(#barGradient)" style="animation-delay: 2.1s" />
                            <rect class="bar bar-glow" x="398" y="186" width="38" height="82" rx="12" fill="url(#barGradient)" style="animation-delay: 2.6s" />
                            <rect class="bar" x="402" y="190" width="30" height="74" rx="12" fill="url(#barGradient)" style="animation-delay: 2.6s" />
                            <rect class="bar-sweep" x="50" y="70" width="620" height="210" fill="url(#sweepGradient)" />
                        </g>
                        <g stroke="rgba(147,197,253,0.5)" stroke-width="1.5">
                            <line x1="120" y1="86" x2="180" y2="120" />
                            <line x1="180" y1="120" x2="240" y2="90" />
                            <line x1="240" y1="90" x2="300" y2="130" />
                            <line x1="300" y1="130" x2="360" y2="100" />
                            <line x1="360" y1="100" x2="420" y2="128" />
                            <line x1="420" y1="128" x2="480" y2="88" />
                        </g>
                        <g>
                            <circle cx="120" cy="86" r="6" fill="url(#nodeGlow)" />
                            <circle cx="180" cy="120" r="6" fill="url(#nodeGlow)" />
                            <circle cx="240" cy="90" r="6" fill="url(#nodeGlow)" />
                            <circle cx="300" cy="130" r="6" fill="url(#nodeGlow)" />
                            <circle cx="360" cy="100" r="6" fill="url(#nodeGlow)" />
                            <circle cx="420" cy="128" r="6" fill="url(#nodeGlow)" />
                            <circle cx="480" cy="88" r="6" fill="url(#nodeGlow)" />
                        </g>
                    </svg>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown('<h3>Welcome Back</h3>', unsafe_allow_html=True)
        st.markdown('<div class="ui-card-body">', unsafe_allow_html=True)
        st.markdown('<p>Login to your account</p>', unsafe_allow_html=True)
        auth_mode = st.session_state.auth_mode

        if auth_mode == 'Login':
            with st.form('login_form'):
                login_username = st.text_input('Username')
                login_password = st.text_input('Password', type='password')
                login_btn = st.form_submit_button('Login', key='login_btn')

            if login_btn:
                user = authenticate_user(login_username, login_password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = user[1]
                    st.session_state.full_name = user[2]
                    st.success(f'Welcome back, {user[2]}!')
                    st.rerun()
                else:
                    st.error('Invalid username or password.')

            st.markdown('<div class="ui-link"><a href="#">Forgot password?</a></div>', unsafe_allow_html=True)
            st.markdown('<div class="ui-card-body-actions">', unsafe_allow_html=True)
            st.markdown('<span>Not a member?</span>', unsafe_allow_html=True)
            if st.button('Register here', key='register_btn', type='secondary'):
                st.session_state.auth_mode = 'Register'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            with st.form('register_form'):
                reg_full_name = st.text_input('Full Name')
                reg_username = st.text_input('Username (4-30 chars, letters/numbers/._-)')
                reg_password = st.text_input('Password (min 8 chars, include letters and numbers)', type='password')
                reg_confirm_password = st.text_input('Confirm Password', type='password')
                reg_btn = st.form_submit_button('Create Account')

            if reg_btn:
                if not reg_full_name.strip():
                    st.error('Full name is required.')
                elif not valid_username(reg_username):
                    st.error('Invalid username format.')
                elif not valid_password(reg_password):
                    st.error('Password must be at least 8 characters and include letters and numbers.')
                elif reg_password != reg_confirm_password:
                    st.error('Passwords do not match.')
                else:
                    ok, msg = create_user(reg_username, reg_full_name, reg_password)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

            st.markdown('<div class="ui-card-body-actions">', unsafe_allow_html=True)
            st.markdown('<span>Already have an account?</span>', unsafe_allow_html=True)
            if st.button('Back to Login', key='login_link', type='secondary'):
                st.session_state.auth_mode = 'Login'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


# Sidebar with project info and file upload
st.sidebar.title('Churn Prediction System')
st.sidebar.info('Upload your customer data and explore churn risk using unsupervised learning and explainable AI.')
st.sidebar.success(f"Logged in as: {st.session_state.full_name}")
if st.sidebar.button('Logout'):
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.full_name = ''
    st.rerun()
st.sidebar.markdown('---')
st.sidebar.header('Upload Data')
uploaded_file = st.sidebar.file_uploader('Upload your customer data (CSV)', type=['csv'])
st.sidebar.markdown('---')
st.sidebar.write('Developed with ❤️ using Streamlit')

st.markdown('## Upload Data')
st.caption('Use the uploader below or the sidebar to load your CSV.')
upload_main = st.file_uploader('Upload your customer data (CSV)', type=['csv'])
if upload_main is not None:
    uploaded_file = upload_main
if st.button('Logout'):
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.full_name = ''
    st.rerun()
st.markdown('---')

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write('### Raw Data', data.head())
    st.write('#### Columns in your data:', list(data.columns))

    required_cols = ['Customer ID', 'InvoiceDate', 'Invoice', 'Quantity', 'Price']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.warning("Please upload a CSV with the required columns or update the code to match your data.")
    else:
        # User selects number of clusters
        n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=8, value=4, help='Choose how many customer segments to create')

        # Tabs for each analysis section
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            'RFM Analysis', 'RFM Distributions', 'Clustering', 'Churn Prediction', 'Explainable AI (SHAP)'])

        with tab1:
            st.write('## RFM Analysis')
            st.info('RFM (Recency, Frequency, Monetary) analysis segments customers based on how recently, how often, and how much they purchase. This helps identify valuable and at-risk customers.')
            data['TotalPrice'] = data['Quantity'] * data['Price']
            # Ensure InvoiceDate is datetime
            data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
            rfm = data.groupby('Customer ID').agg({
                'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,
                'Invoice': 'nunique',
                'TotalPrice': 'sum'
            })
            rfm.columns = ['Recency', 'Frequency', 'Monetary']
            st.dataframe(rfm.head())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Total Customers', len(rfm))
            c2.metric('Avg Recency', int(rfm['Recency'].mean()))
            c3.metric('Avg Frequency', round(rfm['Frequency'].mean(), 2))
            c4.metric('Avg Monetary', round(rfm['Monetary'].mean(), 2))

        with tab2:
            st.write('### RFM Feature Distributions')
            st.caption('These histograms show the distribution of Recency, Frequency, and Monetary values across all customers.')
            rfm_cols = st.columns(3)
            with rfm_cols[0]:
                fig_rfm_recency, ax_rfm_recency = plt.subplots(figsize=(4.4, 3.2))
                ax_rfm_recency.hist(rfm['Recency'], bins=20, color='skyblue')
                ax_rfm_recency.set_title('Recency')
                st.pyplot(fig_rfm_recency, use_container_width=False)
            with rfm_cols[1]:
                fig_rfm_frequency, ax_rfm_frequency = plt.subplots(figsize=(4.4, 3.2))
                ax_rfm_frequency.hist(rfm['Frequency'], bins=20, color='lightgreen')
                ax_rfm_frequency.set_title('Frequency')
                st.pyplot(fig_rfm_frequency, use_container_width=False)
            with rfm_cols[2]:
                fig_rfm_monetary, ax_rfm_monetary = plt.subplots(figsize=(4.4, 3.2))
                ax_rfm_monetary.hist(rfm['Monetary'], bins=20, color='salmon')
                ax_rfm_monetary.set_title('Monetary')
                st.pyplot(fig_rfm_monetary, use_container_width=False)
            st.markdown(
                "- **Recency**: Lower values mean recent purchases; higher values indicate longer time since last purchase.\n"
                "- **Frequency**: Higher values mean more invoices (more frequent purchases).\n"
                "- **Monetary**: Higher values indicate greater total spending. These shapes show how customers are distributed across each RFM dimension."
            )

        with tab3:
            st.write('## Hybrid Clustering')
            st.info('Clustering groups customers with similar RFM profiles. KMeans and Agglomerative clustering are used to find natural segments in your customer base.')
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(rfm_scaled)
            agg = AgglomerativeClustering(n_clusters=n_clusters)
            agg_labels = agg.fit_predict(rfm_scaled)
            rfm['KMeans_Cluster'] = kmeans_labels
            rfm['Agg_Cluster'] = agg_labels
            st.dataframe(rfm.head())
            st.write('### Cluster Scatterplot (Recency vs Monetary)')
            st.caption('Each point is a customer, colored by their cluster. This helps visualize how clusters separate based on Recency and Monetary value.')
            fig_scatter, ax_scatter = plt.subplots(figsize=(6.5, 4.2))
            scatter = ax_scatter.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['KMeans_Cluster'], cmap='tab10', alpha=0.7)
            legend1 = ax_scatter.legend(*scatter.legend_elements(), title="KMeans Cluster")
            ax_scatter.add_artist(legend1)
            ax_scatter.set_xlabel('Recency')
            ax_scatter.set_ylabel('Monetary')
            st.pyplot(fig_scatter, use_container_width=False)
            st.markdown(
                "This scatter shows customers by **Recency** (x-axis) and **Monetary** (y-axis), colored by KMeans cluster.\n"
                "- Points to the right (higher Recency) are less recent customers.\n"
                "- Points lower (smaller Monetary) spend less.\n"
                "Distinct color groups indicate segments with similar RFM behavior."
            )

        with tab4:
            st.write('## Churn Prediction (Cluster-based)')
            st.info('Customers in certain clusters (e.g., high Recency, low Frequency/Monetary) may be at higher risk of churn. This unsupervised approach uses clusters as a proxy for churn risk.')
            cluster_counts = rfm['KMeans_Cluster'].value_counts().sort_index()
            st.write('### Number of Customers per KMeans Cluster')
            st.caption('This bar chart shows how many customers are in each cluster.')
            fig_bar, ax_bar = plt.subplots(figsize=(6.2, 3.8))
            ax_bar.bar(cluster_counts.index.astype(str), cluster_counts.values, color='orchid')
            ax_bar.set_xlabel('KMeans Cluster')
            ax_bar.set_ylabel('Number of Customers')
            st.pyplot(fig_bar, use_container_width=False)
            st.markdown(
                "This bar chart shows how many customers fall into each KMeans cluster.\n"
                "Use this to gauge the size of segments (e.g., whether the high-risk cluster is small and targeted or large and widespread)."
            )
            st.write(rfm.groupby('KMeans_Cluster').mean())

            cluster_profile = rfm.groupby('KMeans_Cluster')[['Recency', 'Frequency', 'Monetary']].mean().copy()
            rec_norm = (cluster_profile['Recency'] - cluster_profile['Recency'].min()) / (
                cluster_profile['Recency'].max() - cluster_profile['Recency'].min() + 1e-9
            )
            freq_norm = (cluster_profile['Frequency'] - cluster_profile['Frequency'].min()) / (
                cluster_profile['Frequency'].max() - cluster_profile['Frequency'].min() + 1e-9
            )
            mon_norm = (cluster_profile['Monetary'] - cluster_profile['Monetary'].min()) / (
                cluster_profile['Monetary'].max() - cluster_profile['Monetary'].min() + 1e-9
            )
            cluster_profile['RiskScore'] = rec_norm + (1 - freq_norm) + (1 - mon_norm)
            high_risk_cluster = int(cluster_profile['RiskScore'].idxmax())
            rfm['Predicted_Churn'] = np.where(rfm['KMeans_Cluster'] == high_risk_cluster, 'High Risk', 'Lower Risk')

            high_risk_customers = rfm[rfm['Predicted_Churn'] == 'High Risk'].copy()
            high_risk_count = len(high_risk_customers)
            total_customers = len(rfm)
            high_risk_pct = (high_risk_count / total_customers * 100) if total_customers else 0

            st.write('### Final Prediction Statement')
            st.success(
                f"Predicted churn type: **customer inactivity churn risk**. "
                f"Based on your uploaded data, cluster **{high_risk_cluster}** is the highest-risk segment. "
                f"Predicted at-risk customers: **{high_risk_count} / {total_customers} ({high_risk_pct:.2f}%)**."
            )
            st.caption(
                'Interpretation: High-risk customers are those with relatively higher recency (longer time since last purchase), '
                'and comparatively lower frequency and/or lower monetary value than other clusters.'
            )

            st.write('### High-Risk Customer List (Top 100)')
            st.dataframe(high_risk_customers[['Recency', 'Frequency', 'Monetary', 'KMeans_Cluster', 'Predicted_Churn']].head(100))

            # Download button for cluster results
            csv = rfm.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button('Download Clustered Data as CSV', csv, 'clustered_customers.csv', 'text/csv')

        with tab5:
            st.write('## Explainable AI (SHAP)')
            st.info('SHAP (SHapley Additive exPlanations) explains which RFM features are most important for assigning customers to clusters, helping you understand the drivers of churn risk.')
            explainer = shap.KernelExplainer(kmeans.predict, rfm_scaled)
            shap_values = explainer.shap_values(rfm_scaled[:50])
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            shap.summary_plot(shap_values, rfm.iloc[:50, :3], show=False)
            st.pyplot(fig, use_container_width=False)
            st.markdown(
                "The SHAP summary ranks features by their contribution to cluster assignment.\n"
                "- Larger absolute SHAP values mean stronger influence.\n"
                "- Color often reflects feature value (depends on SHAP plot style).\n"
                "Use this to understand whether **Recency**, **Frequency**, or **Monetary** drives segmentation and churn risk."
            )

        st.write('---')
        st.write('This is a demo. For production, tune clustering and RFM logic to your data.')
else:
    st.markdown('<script>document.body.classList.remove("auth-mode");</script>', unsafe_allow_html=True)
    st.info('Awaiting CSV file upload.')
