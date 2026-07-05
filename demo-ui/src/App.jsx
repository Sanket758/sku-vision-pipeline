import React, { useState, useEffect } from 'react';

const experiments = [
  { id: 'EXP-042', model: 'moondream:latest + CLIP', dataset: 'Supermarket_Shelf_V2', precision: '92.4%', inference: '240ms', vram: '4.2 GB', status: 'Completed', statusClass: 'status-success' },
  { id: 'EXP-043', model: 'llava:7b + CLIP', dataset: 'Supermarket_Shelf_V2', precision: '94.1%', inference: '850ms', vram: '6.8 GB (OOM)', status: 'Failed', statusClass: 'status-down' },
  { id: 'EXP-044', model: 'moondream:1.8b (Zero-Shot)', dataset: 'Supermarket_Shelf_V2', precision: '88.5%', inference: '210ms', vram: '3.8 GB', status: 'Completed', statusClass: 'status-success' },
  { id: 'EXP-045', model: 'moondream:latest + FAISS', dataset: 'Supermarket_Shelf_V3_Dark', precision: '-', inference: '-', vram: '4.5 GB', status: 'Running', statusClass: 'status-running' }
];

const mockDetections = [
  { label: 'Dallmayr Prodomo', top: '26.2%', left: '36.5%', width: '13%', height: '8%' },
  { label: 'Jacobs Krönung', top: '26.2%', left: '7.5%', width: '18%', height: '8%' },
  { label: 'Barilla Penne', top: '47.5%', left: '27.5%', width: '10.5%', height: '7%' }
];

export default function App() {
  const [mounted, setMounted] = useState(false);
  
  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar glass-panel">
        <div className="logo-area">
          <div className="logo-icon">T</div>
          ThesisOps UI
        </div>
        
        <nav className="nav-menu">
          <a href="#" className="nav-item active">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
            Dashboard
          </a>
          <a href="#" className="nav-item">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
            Experiments
          </a>
          <a href="#" className="nav-item">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
            Models
          </a>
          <a href="#" className="nav-item">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
            Datasets
          </a>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="header glass-panel">
          <div className="header-title">
            <h1>Product Recognition Tracker</h1>
            <p>Chapter 4 - Methodology Experiments & Vision Agent Analysis</p>
          </div>
          <div style={{display: 'flex', gap: '12px'}}>
            <button style={{background: 'var(--surface-color)', border: '1px solid var(--surface-border)', color: '#fff', padding: '8px 16px', borderRadius: '8px', cursor: 'pointer', transition: 'all 0.2s'}} onMouseOver={(e) => e.target.style.background = 'rgba(255,255,255,0.1)'} onMouseOut={(e) => e.target.style.background = 'var(--surface-color)'}>Export Report</button>
            <button style={{background: 'linear-gradient(135deg, var(--accent-color), #00b4d8)', border: 'none', color: '#000', fontWeight: 'bold', padding: '8px 16px', borderRadius: '8px', cursor: 'pointer', transition: 'transform 0.2s', boxShadow: '0 4px 15px rgba(69, 243, 255, 0.3)'}} onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'} onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}>Run Agent</button>
          </div>
        </header>

        <div className="metrics-row">
          <div className="metric-card glass-panel" style={{ opacity: mounted ? 1 : 0, transitionDelay: '100ms' }}>
            <span className="metric-title">Best Precision (moondream)</span>
            <span className="metric-value">92.4%</span>
            <span className="metric-trend trend-up">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
              +3.2% vs baseline
            </span>
          </div>
          <div className="metric-card glass-panel" style={{ opacity: mounted ? 1 : 0, transitionDelay: '200ms' }}>
            <span className="metric-title">Avg Inference Time</span>
            <span className="metric-value">240ms</span>
            <span className="metric-trend trend-up">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
              Highly Real-time
            </span>
          </div>
          <div className="metric-card glass-panel" style={{ opacity: mounted ? 1 : 0, transitionDelay: '300ms' }}>
            <span className="metric-title">Peak VRAM Usage</span>
            <span className="metric-value">4.5 GB</span>
            <span className="metric-trend trend-up">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
              Safely under 6GB limit
            </span>
          </div>
        </div>

        <div className="dashboard-grid">
          {/* Demo Viewer */}
          <div className="demo-viewer glass-panel" style={{ opacity: mounted ? 1 : 0, transitionDelay: '400ms' }}>
            <h3 style={{marginBottom: '8px', color: 'var(--accent-color)'}}>Live Vision Agent Demo</h3>
            <p style={{color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '16px'}}>Real-time product detection using Moondream + CLIP FAISS index.</p>
            
            <div className="image-container">
              <div className="image-wrapper">
                <img src="/supermarket_shelf.png" alt="Supermarket Shelf" />
                
                {/* Dynamic Bounding Boxes */}
                {mockDetections.map((det, idx) => (
                  <div key={idx} className="bbox" style={{ top: det.top, left: det.left, width: det.width, height: det.height, animationDelay: `${idx * 0.3}s` }}>
                    <div className="bbox-label">{det.label}</div>
                  </div>
                ))}
              </div>
            </div>
            
            <div style={{marginTop: '16px', background: 'rgba(0,0,0,0.3)', padding: '12px', borderRadius: '8px', borderLeft: '3px solid var(--accent-secondary)'}}>
              <span style={{color: 'var(--accent-secondary)', fontSize: '0.8rem', fontWeight: 'bold'}}>AGENT LOG:</span>
              <span style={{color: '#fff', fontSize: '0.85rem', marginLeft: '8px', fontFamily: 'monospace'}}>Identified 3 target products with &gt;90% confidence. FAISS similarity match successful.</span>
            </div>
          </div>

          {/* Experiments Table */}
          <div className="experiments-table glass-panel" style={{ opacity: mounted ? 1 : 0, transitionDelay: '500ms' }}>
            <h3 style={{marginBottom: '24px'}}>Recent Experiment Runs</h3>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Model Config</th>
                  <th>Precision</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((exp, idx) => (
                  <tr key={idx}>
                    <td style={{fontWeight: 'bold', color: 'var(--text-muted)'}}>{exp.id}</td>
                    <td>{exp.model}</td>
                    <td>{exp.precision}</td>
                    <td><span className={`status-badge ${exp.statusClass}`}>{exp.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
