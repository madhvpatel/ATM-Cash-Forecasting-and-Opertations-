#!/usr/bin/env python3
"""Transform live_portal/index.html to a light professional theme (Stripe/Linear/Mercury style)."""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
INDEX_HTML = HERE / "index.html"

content = INDEX_HTML.read_text(encoding="utf-8")

# 1. Replace CSS :root and styles up to </style>
LIGHT_CSS = """    :root {
      --bg-base: #f8fafc;
      --bg-sidebar: #ffffff;
      --bg-surface: #ffffff;
      --bg-surface-elevated: #f1f5f9;
      --bg-card: #ffffff;
      --bg-card-hover: #f8fafc;
      --border-subtle: #e2e8f0;
      --border-focus: rgba(37, 99, 235, 0.45);
      
      --text-main: #0f172a;
      --text-muted: #64748b;
      --text-dim: #94a3b8;
      
      --primary: #2563eb;
      --primary-glow: rgba(37, 99, 235, 0.12);
      --emerald: #059669;
      --emerald-glow: rgba(5, 150, 105, 0.12);
      --amber: #d97706;
      --amber-glow: rgba(217, 119, 6, 0.12);
      --rose: #e11d48;
      --rose-glow: rgba(225, 29, 72, 0.14);
      --purple: #7c3aed;
      --purple-glow: rgba(124, 58, 237, 0.12);
      --indigo: #4f46e5;
      --cyan: #0284c7;

      --font-sans: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-sans);
      line-height: 1.5;
      min-height: 100vh;
      display: flex;
      overflow-x: hidden;
    }

    /* Ambient background lighting */
    .glow-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: radial-gradient(circle at 12% 12%, rgba(37, 99, 235, 0.03), transparent 45%),
                  radial-gradient(circle at 88% 18%, rgba(5, 150, 105, 0.03), transparent 45%),
                  radial-gradient(circle at 50% 90%, rgba(79, 70, 229, 0.02), transparent 50%);
      pointer-events: none;
      z-index: 0;
    }

    /* --- SIDEBAR NAVIGATION --- */
    .sidebar {
      width: 275px;
      min-width: 275px;
      background: var(--bg-sidebar);
      border-right: 1px solid var(--border-subtle);
      display: flex;
      flex-direction: column;
      position: sticky;
      top: 0;
      height: 100vh;
      z-index: 100;
      box-shadow: 1px 0 4px rgba(0, 0, 0, 0.03);
      user-select: none;
    }

    .sidebar-header {
      padding: 20px 22px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      gap: 12px;
      background: #ffffff;
    }

    .brand-logo {
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, #0284c7, #2563eb);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 3px 10px rgba(37, 99, 235, 0.25);
      flex-shrink: 0;
    }

    .brand-logo svg { width: 22px; height: 22px; color: #fff; }

    .brand-text h1 {
      font-size: 1.05rem;
      font-weight: 800;
      color: var(--text-main);
      letter-spacing: -0.02em;
      line-height: 1.2;
    }

    .brand-text span {
      font-size: 0.72rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }

    .sidebar-nav {
      flex: 1;
      overflow-y: auto;
      padding: 16px 14px;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }

    .nav-group-title {
      font-size: 0.68rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--text-muted);
      padding: 0 10px 6px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .nav-items {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }

    .nav-btn {
      display: flex;
      align-items: center;
      gap: 11px;
      padding: 9px 12px;
      border-radius: 9px;
      color: #475569;
      font-size: 0.83rem;
      font-weight: 600;
      text-decoration: none;
      background: transparent;
      border: 1px solid transparent;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      text-align: left;
      width: 100%;
      position: relative;
    }

    .nav-btn:hover {
      color: #0f172a;
      background: #f1f5f9;
      border-color: #e2e8f0;
    }

    .nav-btn.active {
      color: #1d4ed8;
      background: #eff6ff;
      border-color: #bfdbfe;
      box-shadow: 0 1px 3px rgba(37, 99, 235, 0.08);
      font-weight: 700;
    }

    .nav-btn.active .nav-icon {
      color: #2563eb;
    }

    .nav-btn.active::before {
      content: '';
      position: absolute;
      left: 0;
      top: 18%;
      bottom: 18%;
      width: 3px;
      background: #2563eb;
      border-radius: 0 4px 4px 0;
      box-shadow: 0 0 8px rgba(37, 99, 235, 0.4);
    }

    .nav-icon {
      font-size: 0.95rem;
      width: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #64748b;
      flex-shrink: 0;
      transition: color 0.2s;
    }

    .nav-step-badge {
      font-family: var(--font-mono);
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      background: #f1f5f9;
      padding: 1px 5px;
      border-radius: 4px;
      border: 1px solid #e2e8f0;
    }

    .nav-btn.active .nav-step-badge {
      color: #1d4ed8;
      background: #dbeafe;
      border-color: #bfdbfe;
    }

    .nav-badge-pill {
      margin-left: auto;
      font-size: 0.68rem;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 9999px;
      font-family: var(--font-mono);
    }

    .badge-rose {
      background: #ffe4e6;
      color: #be123c;
      border: 1px solid #fecdd3;
    }

    .badge-amber {
      background: #fef3c7;
      color: #b45309;
      border: 1px solid #fde68a;
    }

    .badge-emerald {
      background: #d1fae5;
      color: #047857;
      border: 1px solid #a7f3d0;
    }

    .badge-blue {
      background: #dbeafe;
      color: #1d4ed8;
      border: 1px solid #bfdbfe;
    }

    .sidebar-footer {
      padding: 16px 18px;
      border-top: 1px solid var(--border-subtle);
      background: #f8fafc;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .role-card {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      background: #ffffff;
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      cursor: pointer;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
      transition: all 0.2s;
    }

    .role-card:hover {
      border-color: #2563eb;
      box-shadow: 0 2px 8px rgba(37, 99, 235, 0.1);
    }

    .role-avatar {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: linear-gradient(135deg, #0284c7, #2563eb);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      font-weight: 800;
      font-size: 0.76rem;
      flex-shrink: 0;
    }

    .role-info {
      flex: 1;
      overflow: hidden;
    }

    .role-name {
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--text-main);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .role-sub {
      font-size: 0.68rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }

    /* --- MAIN CONTENT LAYOUT --- */
    .app-main {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
      overflow-x: hidden;
      position: relative;
      z-index: 10;
    }

    /* Header Bar */
    .top-bar {
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 12px 28px;
      position: sticky;
      top: 0;
      z-index: 80;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    .top-left {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .screen-title-wrap h2 {
      font-size: 1.15rem;
      font-weight: 800;
      color: var(--text-main);
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .screen-title-wrap p {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    .top-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .meta-chip {
      background: #f1f5f9;
      border: 1px solid var(--border-subtle);
      border-radius: 9999px;
      padding: 5px 12px;
      font-size: 0.75rem;
      color: #475569;
      display: flex;
      align-items: center;
      gap: 7px;
      font-family: var(--font-mono);
    }

    .meta-chip strong {
      color: var(--text-main);
    }

    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--emerald);
      box-shadow: 0 0 6px rgba(5, 150, 105, 0.4);
    }

    .btn-action {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      color: #334155;
      padding: 7px 14px;
      border-radius: 9px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
      transition: all 0.2s;
    }

    .btn-action:hover {
      border-color: #94a3b8;
      color: #0f172a;
      background: #f8fafc;
    }

    .btn-action.active {
      background: #eff6ff;
      border-color: #3b82f6;
      color: #1d4ed8;
      font-weight: 700;
    }

    .btn-primary {
      background: #2563eb;
      border: 1px solid #1d4ed8;
      color: #fff;
      box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
    }

    .btn-primary:hover {
      background: #1d4ed8;
      border-color: #1e40af;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
      color: #fff;
    }

    .btn-emerald {
      background: #059669;
      border: 1px solid #047857;
      color: #fff;
      box-shadow: 0 2px 6px rgba(5, 150, 105, 0.25);
    }

    .btn-emerald:hover {
      background: #047857;
      border-color: #065f46;
      box-shadow: 0 4px 12px rgba(5, 150, 105, 0.35);
      color: #fff;
    }

    /* --- OPERATIONAL WORKFLOW STEPPER BAR --- */
    .workflow-stepper-bar {
      background: #ffffff;
      border-bottom: 1px solid var(--border-subtle);
      padding: 12px 28px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .stepper-progress-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: relative;
    }

    .stepper-progress-row::before {
      content: '';
      position: absolute;
      top: 50%;
      left: 30px;
      right: 30px;
      height: 2px;
      background: #e2e8f0;
      transform: translateY(-50%);
      z-index: 1;
    }

    .step-node {
      display: flex;
      align-items: center;
      gap: 8px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 30px;
      padding: 6px 14px;
      font-size: 0.78rem;
      font-weight: 700;
      color: #64748b;
      cursor: pointer;
      position: relative;
      z-index: 2;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .step-node:hover {
      color: #0f172a;
      border-color: #cbd5e1;
      background: #ffffff;
    }

    .step-node.active {
      color: #1d4ed8;
      background: #eff6ff;
      border-color: #3b82f6;
      box-shadow: 0 2px 8px rgba(37, 99, 235, 0.15);
    }

    .step-node.completed {
      color: #047857;
      border-color: #a7f3d0;
      background: #ecfdf5;
    }

    .step-num {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: #e2e8f0;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.7rem;
      font-family: var(--font-mono);
      font-weight: 800;
      color: #475569;
    }

    .step-node.active .step-num {
      background: #2563eb;
      color: #fff;
    }

    .step-node.completed .step-num {
      background: #059669;
      color: #fff;
    }

    /* Stepper Context Chip */
    .stepper-context-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #f0f9ff;
      border: 1px solid #bae6fd;
      border-radius: 8px;
      padding: 7px 14px;
      font-size: 0.78rem;
    }

    .context-left {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    /* --- CONTENT CONTAINER & SCREEN SWITCHING --- */
    .view-container {
      padding: 24px 28px;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 24px;
      max-width: 1560px;
      width: 100%;
      margin: 0 auto;
    }

    .portal-screen {
      display: none;
      flex-direction: column;
      gap: 22px;
      animation: fadeIn 0.25s ease-out;
    }

    .portal-screen.active {
      display: flex;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Glass Cards */
    .glass-card {
      background: #ffffff;
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 20px 24px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 16px;
      gap: 14px;
      flex-wrap: wrap;
    }

    .card-title h3 {
      font-size: 1.05rem;
      font-weight: 800;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .card-title p {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 2px;
    }

    /* KPI Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
    }

    @media (max-width: 1200px) {
      .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 680px) {
      .kpi-grid { grid-template-columns: 1fr; }
    }

    .kpi-card {
      background: #ffffff;
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .kpi-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
    }

    .kpi-card.cyan::before { background: var(--primary); }
    .kpi-card.emerald::before { background: var(--emerald); }
    .kpi-card.amber::before { background: var(--amber); }
    .kpi-card.rose::before { background: var(--rose); }
    .kpi-card.purple::before { background: var(--purple); }

    .kpi-title {
      font-size: 0.76rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .kpi-val {
      font-family: var(--font-mono);
      font-size: 1.55rem;
      font-weight: 800;
      color: var(--text-main);
      line-height: 1.1;
    }

    .kpi-sub {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    /* Grids */
    .two-col-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }

    .three-col-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
    }

    @media (max-width: 1050px) {
      .two-col-grid, .three-col-grid { grid-template-columns: 1fr; }
    }

    /* Tag styles */
    .tag {
      font-size: 0.72rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-family: var(--font-mono);
    }

    .tag-rose { background: #ffe4e6; color: #be123c; border: 1px solid #fecdd3; }
    .tag-amber { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .tag-emerald { background: #d1fae5; color: #047857; border: 1px solid #a7f3d0; }
    .tag-primary { background: #dbeafe; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .tag-purple { background: #ede9fe; color: #6d28d9; border: 1px solid #ddd6fe; }
    .tag-dim { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }

    /* Tables */
    .portal-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.82rem;
    }

    .portal-table th {
      background: #f8fafc;
      color: #475569;
      font-weight: 700;
      text-transform: uppercase;
      font-size: 0.7rem;
      letter-spacing: 0.05em;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-subtle);
      text-align: left;
    }

    .portal-table td {
      padding: 12px 14px;
      border-bottom: 1px solid #f1f5f9;
      color: var(--text-main);
      vertical-align: middle;
      background: #ffffff;
    }

    .portal-table tr:hover td {
      background: #f8fafc;
    }

    .portal-table tr.selected td {
      background: #eff6ff;
      border-color: #bfdbfe;
    }

    /* Exception Callout Banner */
    .exception-banner {
      background: #fff1f2;
      border: 1px solid #fecdd3;
      border-radius: 12px;
      padding: 16px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }

    .exception-left {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .exception-icon {
      width: 38px;
      height: 38px;
      border-radius: 10px;
      background: #ffe4e6;
      border: 1px solid #fecdd3;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
      color: #be123c;
      flex-shrink: 0;
    }

    /* Transparent Indent Math Box */
    .indent-math-box {
      background: #f8fafc;
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 18px 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .equation-display {
      font-family: var(--font-mono);
      font-size: 0.95rem;
      color: #0369a1;
      background: #ffffff;
      padding: 10px 16px;
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
      border-left: 4px solid var(--primary);
    }

    .math-breakdown-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
    }

    @media (max-width: 900px) {
      .math-breakdown-grid { grid-template-columns: repeat(2, 1fr); }
    }

    .math-step {
      background: #ffffff;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    .math-step-label {
      font-size: 0.7rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 700;
    }

    .math-step-val {
      font-family: var(--font-mono);
      font-size: 1.05rem;
      font-weight: 800;
      color: var(--text-main);
    }

    /* CIT Execution Pipeline */
    .pipeline-stepper {
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: relative;
      padding: 10px 0;
    }

    .pipeline-stepper::before {
      content: '';
      position: absolute;
      top: 50%;
      left: 20px;
      right: 20px;
      height: 2px;
      background: #e2e8f0;
      transform: translateY(-50%);
      z-index: 1;
    }

    .pipe-node {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      position: relative;
      z-index: 2;
      background: #ffffff;
      padding: 0 10px;
    }

    .pipe-circle {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: #f1f5f9;
      border: 2px solid #cbd5e1;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      font-weight: 800;
      color: #64748b;
    }

    .pipe-node.done .pipe-circle {
      background: #059669;
      border-color: #059669;
      color: #ffffff;
    }

    .pipe-node.current .pipe-circle {
      background: #2563eb;
      border-color: #2563eb;
      color: #ffffff;
      box-shadow: 0 0 10px rgba(37, 99, 235, 0.35);
    }

    .pipe-label {
      font-size: 0.72rem;
      font-weight: 700;
      color: var(--text-muted);
      text-align: center;
      white-space: nowrap;
    }

    .pipe-node.current .pipe-label { color: #2563eb; }
    .pipe-node.done .pipe-label { color: #059669; }

    /* Interactive Inputs & Sliders */
    .input-field {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      color: var(--text-main);
      font-family: var(--font-sans);
      font-size: 0.88rem;
      padding: 8px 12px;
      border-radius: 8px;
      outline: none;
      transition: all 0.2s;
      width: 100%;
    }

    .input-field:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }

    .select-field {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      color: var(--text-main);
      font-family: var(--font-sans);
      font-size: 0.84rem;
      padding: 8px 12px;
      border-radius: 8px;
      outline: none;
      cursor: pointer;
      width: 100%;
    }

    .select-field:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }

    /* Modals */
    .modal-backdrop {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(15, 23, 42, 0.5);
      backdrop-filter: blur(8px);
      z-index: 1000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }

    .modal-box {
      background: #ffffff;
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      width: 100%;
      max-width: 650px;
      box-shadow: 0 20px 45px rgba(15, 23, 42, 0.2);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      animation: modalPop 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes modalPop {
      from { opacity: 0; transform: scale(0.96); }
      to { opacity: 1; transform: scale(1); }
    }

    .modal-head {
      padding: 18px 24px;
      background: #f8fafc;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .modal-body {
      padding: 22px 24px;
      overflow-y: auto;
      max-height: 80vh;
      display: flex;
      flex-direction: column;
      gap: 16px;
      background: #ffffff;
    }

    .modal-foot {
      padding: 14px 24px;
      border-top: 1px solid var(--border-subtle);
      background: #f8fafc;
      display: flex;
      justify-content: flex-end;
      gap: 10px;
    }

    /* Simulation Lab Sandbox Banner */
    .sandbox-banner {
      background: #f5f3ff;
      border: 1px solid #ddd6fe;
      border-radius: 12px;
      padding: 16px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }

    /* Map Elements */
    #craMap {
      width: 100%;
      height: 520px;
      border-radius: 12px;
      background: #f1f5f9;
      border: 1px solid var(--border-subtle);
    }

    .map-marker {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 0.72rem;
      color: #fff;
      border: 2px solid #fff;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
      cursor: pointer;
    }

    .map-marker-red { background: #e11d48; box-shadow: 0 0 10px rgba(225, 29, 72, 0.6); }
    .map-marker-amber { background: #d97706; box-shadow: 0 0 10px rgba(217, 119, 6, 0.6); }
    .map-marker-green { background: #059669; box-shadow: 0 0 10px rgba(5, 150, 105, 0.6); }

    /* Footer */
    footer {
      border-top: 1px solid var(--border-subtle);
      padding: 16px 28px;
      font-size: 0.75rem;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #ffffff;
    }

    /* Toast Notification System */
    .toast-container {
      position: fixed;
      top: 24px;
      right: 24px;
      z-index: 10000;
      display: flex;
      flex-direction: column;
      gap: 10px;
      pointer-events: none;
    }

    .toast-msg {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      padding: 14px 18px;
      border-radius: 12px;
      box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12), 0 2px 6px rgba(15, 23, 42, 0.06);
      display: flex;
      align-items: flex-start;
      gap: 12px;
      min-width: 320px;
      max-width: 440px;
      color: #0f172a;
      pointer-events: auto;
      animation: slideInRight 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      transition: all 0.3s ease;
    }

    @keyframes slideInRight {
      from { transform: translateX(40px); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }

    .toast-msg.toast-success { border-color: #a7f3d0; background: #f0fdf4; }
    .toast-msg.toast-warning { border-color: #fde68a; background: #fffbeb; }
    .toast-msg.toast-error { border-color: #fecdd3; background: #fff1f2; }
    .toast-msg.toast-info { border-color: #bfdbfe; background: #eff6ff; }
  </style>"""

# Replace <style>...</style>
style_pattern = re.compile(r"<style>.*?</style>", re.DOTALL)
content = style_pattern.sub(f"<style>\n{LIGHT_CSS}\n  </style>", content, count=1)

# In the HTML body, replace text styles:
# 1. <strong style="color: #fff;"> -> <strong style="color: var(--text-main);">
content = content.replace('style="color: #fff;"', 'style="color: var(--text-main);"')
content = content.replace("style='color: #fff;'", "style='color: var(--text-main);'")
content = content.replace('color: #fff;', 'color: var(--text-main);')

# Fix specific buttons / logos / avatars that should keep white text:
content = content.replace('.brand-logo svg { width: 22px; height: 22px; color: var(--text-main); }', '.brand-logo svg { width: 22px; height: 22px; color: #fff; }')
content = content.replace('.btn-primary {\n      background: #2563eb;\n      border: 1px solid #1d4ed8;\n      color: var(--text-main);', '.btn-primary {\n      background: #2563eb;\n      border: 1px solid #1d4ed8;\n      color: #fff;')
content = content.replace('.btn-emerald {\n      background: #059669;\n      border: 1px solid #047857;\n      color: var(--text-main);', '.btn-emerald {\n      background: #059669;\n      border: 1px solid #047857;\n      color: #fff;')
content = content.replace('.role-avatar {\n      width: 32px;\n      height: 32px;\n      border-radius: 8px;\n      background: linear-gradient(135deg, #0284c7, #2563eb);\n      display: flex;\n      align-items: center;\n      justify-content: center;\n      color: var(--text-main);', '.role-avatar {\n      width: 32px;\n      height: 32px;\n      border-radius: 8px;\n      background: linear-gradient(135deg, #0284c7, #2563eb);\n      display: flex;\n      align-items: center;\n      justify-content: center;\n      color: #fff;')
content = content.replace('.step-node.active .step-num {\n      background: #2563eb;\n      color: var(--text-main);', '.step-node.active .step-num {\n      background: #2563eb;\n      color: #fff;')
content = content.replace('.step-node.completed .step-num {\n      background: #059669;\n      color: var(--text-main);', '.step-node.completed .step-num {\n      background: #059669;\n      color: #fff;')
content = content.replace('.pipe-node.done .pipe-circle {\n      background: #059669;\n      border-color: #059669;\n      color: var(--text-main);', '.pipe-node.done .pipe-circle {\n      background: #059669;\n      border-color: #059669;\n      color: #ffffff;')
content = content.replace('.pipe-node.current .pipe-circle {\n      background: #2563eb;\n      border-color: #2563eb;\n      color: var(--text-main);', '.pipe-node.current .pipe-circle {\n      background: #2563eb;\n      border-color: #2563eb;\n      color: #ffffff;')
content = content.replace('.map-marker {\n      width: 28px;\n      height: 28px;\n      border-radius: 50%;\n      display: flex;\n      align-items: center;\n      justify-content: center;\n      font-weight: 800;\n      font-size: 0.72rem;\n      color: var(--text-main);', '.map-marker {\n      width: 28px;\n      height: 28px;\n      border-radius: 50%;\n      display: flex;\n      align-items: center;\n      justify-content: center;\n      font-weight: 800;\n      font-size: 0.72rem;\n      color: #fff;')

# Replace dark container backgrounds in HTML:
content = content.replace('background: var(--bg-surface);', 'background: #ffffff;')
content = content.replace('background: rgba(0, 0, 0, 0.25);', 'background: #f8fafc; border: 1px solid var(--border-subtle);')
content = content.replace('background: rgba(0, 0, 0, 0.2);', 'background: #f8fafc; border: 1px solid var(--border-subtle);')
content = content.replace('background: rgba(255, 255, 255, 0.03);', 'background: #f8fafc; border: 1px solid var(--border-subtle);')
content = content.replace('background: rgba(255, 255, 255, 0.02);', 'background: #f8fafc; border: 1px solid var(--border-subtle);')

# In Leaflet Map, make Voyager or Positron the default basemap:
content = content.replace('darkMatter.addTo(leafletMapInstance);', 'voyager.addTo(leafletMapInstance);')

# In Chart.js scales, update grid lines & tick colors for light mode:
content = content.replace("grid: { color: 'rgba(255, 255, 255, 0.05)' }", "grid: { color: 'rgba(0, 0, 0, 0.06)' }")
content = content.replace("ticks: { color: '#64748b' }", "ticks: { color: '#475569', font: { weight: 500 } }")
content = content.replace("legend: { labels: { color: '#94a3b8'", "legend: { labels: { color: '#334155', font: { weight: 600 }")

# Write updated file
INDEX_HTML.write_text(content, encoding="utf-8")
print(f"Updated {INDEX_HTML} to lighter professional theme successfully!")
