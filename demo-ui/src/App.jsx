import { useState, useRef, useCallback } from 'react';
import detectionData from './detections.js';
import './App.css';

const { images, metrics, experiments } = detectionData;

/* ─── small helpers ─── */
const pct = (v) => `${v}%`;
const confPct = (v) => `${Math.round(v * 100)}%`;

/* ─── sidebar nav sections ─── */
const SECTIONS = [
  { id: 'gallery', label: 'Gallery', icon: 'M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z' },
  { id: 'results', label: 'Results', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
  { id: 'about', label: 'About', icon: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' },
];

/* ─── confidence → color ─── */
const confColor = (c) => {
  if (c >= 0.9) return '#4ade80';
  if (c >= 0.7) return '#6c8cff';
  if (c >= 0.5) return '#e8a87c';
  return '#f87171';
};

const hexToRgba = (hex, alpha) => {
  const v = parseInt(hex.slice(1), 16);
  return `rgba(${(v >> 16) & 255}, ${(v >> 8) & 255}, ${v & 255}, ${alpha})`;
};

/* ─── Experiment metric helpers ─── */
const getMetric = (exp) => {
  if (exp.f1 != null) return `F1 ${(exp.f1 * 100).toFixed(1)}%`;
  if (exp.accuracy != null) return `Acc ${(exp.accuracy * 100).toFixed(1)}%`;
  if (exp.top1_acc != null) return `Top-1 ${(exp.top1_acc * 100).toFixed(1)}%`;
  if (exp.precision != null) return `Prec ${(exp.precision * 100).toFixed(1)}%`;
  return '—';
};

const getLatency = (exp) => {
  if (exp.avg_latency_s != null) return `${exp.avg_latency_s}s`;
  return '—';
};

/* ════════════════════════════════════════
   MAIN APP
   ════════════════════════════════════════ */
export default function App() {
  const [selectedImage, setSelectedImage] = useState(0);
  const [hoveredDet, setHoveredDet] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [activeSection, setActiveSection] = useState('gallery');
  const viewerRef = useRef(null);
  const resultsRef = useRef(null);
  const aboutRef = useRef(null);

  const currentImage = images[selectedImage];
  const sectionRefs = { gallery: viewerRef, results: resultsRef, about: aboutRef };

  const scrollTo = useCallback((id) => {
    setActiveSection(id);
    const ref = sectionRefs[id];
    if (ref?.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, []);

  const handleDetHover = useCallback((det, e) => {
    setHoveredDet(det);
    setTooltipPos({ x: e.clientX, y: e.clientY });
  }, []);

  const handleDetMove = useCallback((e) => {
    if (hoveredDet) {
      setTooltipPos({ x: e.clientX, y: e.clientY });
    }
  }, [hoveredDet]);

  const handleDetLeave = useCallback(() => {
    setHoveredDet(null);
  }, []);

  /* ─── render ─── */
  return (
    <div className="app">
      {/* ═══ Sidebar ═══ */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#6c8cff" strokeWidth="2">
              <path d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/>
            </svg>
          </div>
          <div className="brand-text">
            <span className="brand-name">SKU Detection</span>
            <span className="brand-sub">Thesis Demo · Ch 4–5</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {SECTIONS.map((s) => (
            <button
              key={s.id}
              className={`nav-btn${activeSection === s.id ? ' active' : ''}`}
              onClick={() => scrollTo(s.id)}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d={s.icon} />
              </svg>
              <span>{s.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-badge">
            <span className="badge-dot" />
            Static Demo
          </div>
          <span className="sidebar-version">v1.0.0</span>
        </div>
      </aside>

      {/* ═══ Main ═══ */}
      <main className="main">
        {/* ─── Metrics Row ─── */}
        <section className="metrics-row">
          <MetricCard
            title="Detection mAP@50"
            value={(metrics.detection_map50 * 100).toFixed(1) + '%'}
            sub="COCO-style evaluation"
            accent="#6c8cff"
            icon="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
          <MetricCard
            title="Recognition Top-1"
            value={(metrics.recognition_top1 * 100).toFixed(1) + '%'}
            sub="DINOv3 + MobileNetV2 hybrid"
            accent="#e8a87c"
            icon="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
          />
          <MetricCard
            title="Total SKUs"
            value={String(metrics.total_skus)}
            sub={`${metrics.total_exemplars} reference exemplars`}
            accent="#4ade80"
            icon="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
          />
          <MetricCard
            title="Pipeline Latency"
            value={`${metrics.pipeline_latency_per_crop_ms}ms`}
            sub="per detected crop"
            accent="#8b8fa3"
            icon="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </section>

        {/* ─── Gallery Section ─── */}
        <section className="section gallery-section" ref={viewerRef} id="section-gallery">
          <div className="section-header">
            <h2 className="section-title">Detection Gallery</h2>
            <span className="section-count">{images.length} shelf scenes</span>
          </div>

          {/* Image viewer */}
          <div className="viewer-panel">
            <div className="image-area">
              <div className="image-container">
                <div className="image-scroller">
                  <img
                    className="shelf-image"
                    src={`/${currentImage.filename}`}
                    alt={currentImage.label}
                    onMouseMove={handleDetMove}
                  />
                  {/* Detection overlays */}
                  {currentImage.detections.map((det, idx) => (
                    <div
                      key={idx}
                      className="det-box"
                      style={{
                        left: pct(det.bbox.x),
                        top: pct(det.bbox.y),
                        width: pct(det.bbox.w),
                        height: pct(det.bbox.h),
                        borderColor: confColor(det.confidence),
                        backgroundColor: hexToRgba(confColor(det.confidence), 0.08),
                      }}
                      onMouseEnter={(e) => handleDetHover(det, e)}
                      onMouseMove={handleDetMove}
                      onMouseLeave={handleDetLeave}
                    >
                      <span className="det-label" style={{ backgroundColor: confColor(det.confidence) }}>
                        {det.sku}
                        <span className="det-conf">{confPct(det.confidence)}</span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Image info bar */}
            <div className="image-info">
              <div className="image-info-left">
                <strong>{currentImage.label}</strong>
                <span className="image-info-source">{currentImage.source}</span>
              </div>
              <span className="image-info-dets">
                {currentImage.detections.length} detections
              </span>
            </div>

            {/* Thumbnail strip */}
            <div className="thumb-strip">
              {images.map((img, idx) => (
                <button
                  key={idx}
                  className={`thumb-btn${idx === selectedImage ? ' active' : ''}`}
                  onClick={() => setSelectedImage(idx)}
                >
                  <img src={`/${img.filename}`} alt={img.label} />
                  <span className="thumb-label">{img.label}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* ─── Results Section ─── */}
        <section className="section results-section" ref={resultsRef} id="section-results">
          <div className="section-header">
            <h2 className="section-title">Experiment Results</h2>
            <span className="section-count">{experiments.length} runs</span>
          </div>

          <div className="table-panel">
            <table className="results-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Model / Config</th>
                  <th>Metric</th>
                  <th>Latency</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((exp) => (
                  <tr key={exp.id}>
                    <td className="cell-id">{exp.id}</td>
                    <td className="cell-name">{exp.name}</td>
                    <td className="cell-metric">{getMetric(exp)}</td>
                    <td className="cell-latency">{getLatency(exp)}</td>
                    <td>
                      <span className={`status-badge status-${exp.status}`}>
                        {exp.status === 'completed' ? '✓ Completed' : '✗ Failed'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* ─── About Section ─── */}
        <section className="section about-section" ref={aboutRef} id="section-about">
          <div className="section-header">
            <h2 className="section-title">About This Demo</h2>
          </div>

          <div className="about-grid">
            <div className="about-card">
              <h3>Automated SKU Detection & Recognition</h3>
              <p>
                This demo showcases a two-stage computer vision pipeline designed for
                automated retail shelf analysis. The system first detects product
                regions using a YOLO-based object detector, then recognises individual
                SKUs via a hybrid retrieval model combining DINOv2 visual features with
                a lightweight MobileNetV2 classifier.
              </p>
            </div>
            <div className="about-card">
              <h3>Pipeline Overview</h3>
              <ol className="pipeline-steps">
                <li><strong>Detection</strong> — YOLOv8n locates product bounding boxes on shelf images</li>
                <li><strong>Crop Extraction</strong> — each detected region is cropped and pre-processed</li>
                <li><strong>Feature Extraction</strong> — DINOv2 self-supervised vision transformer encodes crops</li>
                <li><strong>SKU Matching</strong> — MobileNetV2 refines top-3 candidates from a 148-SKU gallery</li>
                <li><strong>Aggregation</strong> — per-shelf results are collated with confidence scores</li>
              </ol>
            </div>
            <div className="about-card">
              <h3>Performance Summary</h3>
              <p>
                The detection stage achieves <strong>{(metrics.detection_map50 * 100).toFixed(1)}% mAP@50</strong>,
                while the hybrid recognition head reaches <strong>{(metrics.recognition_top1 * 100).toFixed(1)}% Top-1
                accuracy</strong> across 148 SKUs with a per-crop latency of
                <strong> {metrics.pipeline_latency_per_crop_ms}ms</strong>, making the pipeline suitable for
                near-real-time retail inventory analysis.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* ═══ Tooltip ═══ */}
      <div
        className={`det-tooltip${hoveredDet ? ' visible' : ''}`}
        style={{
          left: tooltipPos.x + 16,
          top: tooltipPos.y - 10,
        }}
      >
        {hoveredDet && (
          <>
            <img
              className="tooltip-thumb"
              src={`/demo/skus/${hoveredDet.sku}/thumb.jpg`}
              alt={hoveredDet.sku}
            />
            <div className="tooltip-info">
              <span className="tooltip-sku">{hoveredDet.sku}</span>
              <span className="tooltip-conf" style={{ color: confColor(hoveredDet.confidence) }}>
                {confPct(hoveredDet.confidence)} confidence
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/* ─── MetricCard sub-component ─── */
function MetricCard({ title, value, sub, accent, icon }) {
  return (
    <div className="metric-card" style={{ '--card-accent': accent }}>
      <div className="metric-icon" style={{ color: accent }}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d={icon} />
        </svg>
      </div>
      <div className="metric-body">
        <span className="metric-title">{title}</span>
        <span className="metric-value" style={{ color: accent }}>{value}</span>
        <span className="metric-sub">{sub}</span>
      </div>
    </div>
  );
}
