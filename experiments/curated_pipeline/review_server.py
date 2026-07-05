"""Visual review server with explicit crop state model.

Each crop has an explicit status: ``pending`` → ``confirmed`` | ``skipped``.
Actions (accept / create / rename / skip) are atomic, single-crop or
explicit-batch.  ``Done`` is a guarded operation that refuses if any crops
are still pending.  After every registry-mutating action, remaining pending
crops are rescored in-browser without a pipeline restart.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import webbrowser
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import numpy as np

from experiments.curated_pipeline import config
from experiments.curated_pipeline.config import CATEGORIES
from experiments.curated_pipeline.pipeline_utils.matching import SKURegistry

logger = logging.getLogger(__name__)


class CropStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SKIPPED = "skipped"


class ReviewSession:
    """Review state with explicit per-crop status tracking.

    Every crop starts ``PENDING`` and moves to ``CONFIRMED`` or ``SKIPPED``
    only through an explicit user action (single-crop or explicit batch).
    ``Done`` is a pure function of current state — it blocks if any pending
    crops exist, and only processes crops already in a terminal state.
    """

    def __init__(
        self,
        title: str,
        items: list[dict],
        crops_dir: str | Path,
        dino_embs: list[np.ndarray],
        mnet_embs: list[np.ndarray],
        registry: SKURegistry,
    ) -> None:
        self.title = title
        self.items = items
        self.crops_dir = Path(crops_dir)
        self.dino_embs = dino_embs
        self.mnet_embs = mnet_embs
        self.registry = registry

        self.crop_status: dict[int, str] = {}
        self.crop_action: dict[int, dict] = {}

        for i in range(len(items)):
            self.crop_status[i] = CropStatus.PENDING

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def pending_indices(self) -> list[int]:
        return [i for i, s in self.crop_status.items() if s == CropStatus.PENDING]

    @property
    def confirmed_indices(self) -> list[int]:
        return [i for i, s in self.crop_status.items() if s == CropStatus.CONFIRMED]

    @property
    def skipped_indices(self) -> list[int]:
        return [i for i, s in self.crop_status.items() if s == CropStatus.SKIPPED]

    @property
    def next_sku(self) -> str:
        return self.registry._next_sku_id()

    @property
    def sku_list(self) -> list[dict]:
        entries = self.registry.get_all_entries()
        result = []
        for sku_id, entry in sorted(entries.items()):
            result.append({
                "sku_id": sku_id,
                "label": entry["label"],
                "category": entry["category"],
                "n_exemplars": len(entry["exemplars"]),
                "anchor_exemplar": entry["exemplars"][0] if entry["exemplars"] else None,
            })
        return result

    # ── Validation ──────────────────────────────────────────────────────────

    def _validate_pending(self, idx: int) -> None:
        if self.crop_status.get(idx, CropStatus.PENDING) != CropStatus.PENDING:
            raise ValueError(f"Crop {idx} is already {self.crop_status[idx]}")

    # ── Single-crop actions ─────────────────────────────────────────────────

    def commit_accept(self, idx: int, sku_id: str) -> dict:
        self._validate_pending(idx)
        det = self.items[idx]
        self.registry.add_exemplar(
            sku_id, det["crop_path"],
            self.dino_embs[idx], self.mnet_embs[idx],
            self.title,
        )
        self.crop_status[idx] = CropStatus.CONFIRMED
        self.crop_action[idx] = {"action": "accept", "sku_id": sku_id}
        logger.info("  \u2713 crop %d \u2192 %s (accepted)", idx, sku_id)
        return {"idx": idx, "action": "accept", "sku_id": sku_id}

    def commit_create(self, idx: int, label: str, category: str) -> dict:
        self._validate_pending(idx)
        det = self.items[idx]
        sku_id = self.registry.create_sku(
            label, category, det["crop_path"],
            self.dino_embs[idx], self.mnet_embs[idx],
            self.title,
        )
        self.crop_status[idx] = CropStatus.CONFIRMED
        self.crop_action[idx] = {
            "action": "new", "sku_id": sku_id,
            "label": label, "category": category,
        }
        logger.info("  \U0001f195 crop %d \u2192 created %s (%s)", idx, sku_id, label)
        return {"idx": idx, "action": "new", "sku_id": sku_id, "label": label}

    def commit_rename(self, idx: int, old_sku_id: str, label: str, category: str) -> dict:
        self._validate_pending(idx)
        det = self.items[idx]
        entry = self.registry._data.get(old_sku_id)
        if entry:
            entry["label"] = label
            entry["category"] = category
        self.registry.add_exemplar(
            old_sku_id, det["crop_path"],
            self.dino_embs[idx], self.mnet_embs[idx],
            self.title,
        )
        self.crop_status[idx] = CropStatus.CONFIRMED
        self.crop_action[idx] = {
            "action": "rename", "sku_id": old_sku_id,
            "label": label, "category": category,
        }
        logger.info("  \u270f\ufe0f crop %d \u2192 %s renamed to '%s'", idx, old_sku_id, label)
        return {"idx": idx, "action": "rename", "sku_id": old_sku_id, "label": label}

    def commit_skip(self, idx: int) -> dict:
        self._validate_pending(idx)
        self.crop_status[idx] = CropStatus.SKIPPED
        self.crop_action[idx] = {"action": "skip"}
        logger.info("  \u23ed\ufe0f crop %d skipped", idx)
        return {"idx": idx, "action": "skip"}

    def commit_reject(self, idx: int) -> dict:
        """Revert a confirmed crop back to pending. Removes exemplar from registry."""
        if self.crop_status.get(idx) != CropStatus.CONFIRMED:
            raise ValueError(
                f"Crop {idx} is not confirmed "
                f"(status={self.crop_status.get(idx, 'unknown')})"
            )

        action_info = self.crop_action.get(idx, {})
        sku_id = action_info.get("sku_id")
        if not sku_id:
            raise ValueError(f"Crop {idx} has no SKU to reject")

        det = self.items[idx]
        self.registry.remove_exemplar(sku_id, det["crop_path"])

        was_new = action_info.get("action") in ("create", "new")
        if was_new:
            entry = self.registry.get_sku_info(sku_id)
            if entry is None or entry["n_exemplars"] == 0:
                self.registry.delete_sku(sku_id)
                ex_dir = config.PIPELINE_DIR / "exemplars" / sku_id
                if ex_dir.exists():
                    shutil.rmtree(ex_dir)
                mr_path = config.PIPELINE_DIR / "master_references" / f"{sku_id}.jpg"
                if mr_path.exists():
                    mr_path.unlink()
                logger.info("  \u21a9 Empty SKU %s deleted (was created & rejected)", sku_id)

        self.crop_status[idx] = CropStatus.PENDING
        self.crop_action.pop(idx, None)
        del self.items[idx]["sku_id"]

        logger.info("  \u21a9 crop %d rejected from %s (back to pending)", idx, sku_id)
        return {"idx": idx, "action": "reject", "sku_id": sku_id}

    # ── Batch actions ───────────────────────────────────────────────────────

    def batch_commit(self, indices: list[int], action: str, **kwargs) -> list[dict]:
        results = []
        for idx in indices:
            if action == "accept":
                results.append(self.commit_accept(idx, kwargs["sku_id"]))
            elif action == "create":
                results.append(
                    self.commit_create(
                        idx, kwargs["label"], kwargs.get("category", "Other"),
                    )
                )
            elif action == "rename":
                results.append(
                    self.commit_rename(
                        idx, kwargs["sku_id"],
                        kwargs["label"], kwargs.get("category", "Other"),
                    )
                )
            elif action == "skip":
                results.append(self.commit_skip(idx))
            elif action == "reject":
                results.append(self.commit_reject(idx))
        logger.info(
            "  \u2192 Batch %s on indices %s (N=%d)", action, indices, len(indices),
        )
        return results

    # ── Live rescoring ──────────────────────────────────────────────────────

    def rescore_pending(self) -> dict[int, dict]:
        """Re-score all pending crops against current registry.

        Uses cached embeddings (numpy dot products — no inference).
        Returns ``{idx: updated_match_info}`` for the browser.
        """
        updates: dict[int, dict] = {}
        for idx in self.pending_indices:
            de = self.dino_embs[idx]
            me = self.mnet_embs[idx]
            candidates = self.registry.match_new_crop(de, me)

            if not candidates:
                updates[idx] = {
                    "sku_id": None, "label": "New SKU", "score": 0.0,
                    "zone": "none", "candidates": [],
                    "dino_score": None, "mnet_score": None, "n_exemplars": 0,
                    "anchor_exemplar": None,
                }
            else:
                top = candidates[0]
                updates[idx] = {
                    "sku_id": top["sku_id"],
                    "label": top["label"],
                    "score": top["score"],
                    "zone": top["zone"],
                    "anchor_exemplar": top.get("anchor_exemplar", ""),
                    "candidates": candidates,
                    "dino_score": top.get("dino_score"),
                    "mnet_score": top.get("mnet_score"),
                    "n_exemplars": top.get("n_exemplars", 0),
                }

            self.items[idx].update(updates[idx])

        logger.info("  \u2192 Rescored %d pending crops", len(updates))
        return updates

    # ── Done ────────────────────────────────────────────────────────────────

    def finalize(self) -> dict[int, dict]:
        pending = self.pending_indices
        if pending:
            raise ValueError(
                f"{len(pending)} crops still pending — "
                f"confirm or skip them before finishing."
            )

        outcomes: dict[int, dict] = {}
        for idx in range(len(self.items)):
            action = self.crop_action.get(idx, {})
            outcomes[idx] = {
                "idx": idx,
                "status": self.crop_status[idx],
                "action": action.get("action", "none"),
                "sku_id": action.get("sku_id"),
                "label": self.items[idx].get("label", ""),
                "dino_score": self.items[idx].get("dino_score"),
                "mnet_score": self.items[idx].get("mnet_score"),
                "crop_file": Path(self.items[idx]["crop_path"]).name,
            }
        return outcomes

    def print_outcome_table(self, outcomes: dict[int, dict]) -> None:
        _b = "\u2500"
        sep = _b * 80
        print(f"\n{sep}")
        print(f"  Per-Crop Outcomes \u2014 {self.title}")
        print(f"{sep}")
        print(f"  {'Crop':<6} {'Status':<12} {'SKU':<10} {'Label':<30} {'DINO':<8} {'MNet':<8}")
        print(f"  {_b*4:<6} {_b*10:<12} {_b*8:<10} {_b*28:<30} {_b*6:<8} {_b*6:<8}")
        for idx in range(len(self.items)):
            o = outcomes[idx]
            dino_s = f"{o['dino_score']:.4f}" if o['dino_score'] is not None else "\u2014"
            mnet_s = f"{o['mnet_score']:.4f}" if o['mnet_score'] is not None else "\u2014"
            sku = o['sku_id'] or "\u2014"
            label = (o['label'] or "\u2014")[:28]
            print(f"  {idx:<6} {o['status']:<12} {sku:<10} {label:<30} {dino_s:<8} {mnet_s:<8}")
        n_confirmed = sum(1 for o in outcomes.values() if o['status'] == 'confirmed')
        n_skipped = sum(1 for o in outcomes.values() if o['status'] == 'skipped')
        print(f"{sep}")
        print(f"  Total: {len(outcomes)} crops \u2014 {n_confirmed} confirmed, {n_skipped} skipped")
        print(f"{sep}\n")


# ── HTTP handler ─────────────────────────────────────────────────────────────

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SKU Review — {{ title }}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f5f5; color: #222; padding: 0; display: flex; height: 100vh; }
  .sidebar { width: 250px; min-width: 250px; background: #fff; border-right: 1px solid #ddd;
             overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 8px; }
  .sidebar h2 { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px;
                margin-bottom: 4px; }
  .sku-ref { display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 6px;
             cursor: pointer; transition: background 0.15s; }
  .sku-ref:hover { background: #f0f0f0; }
  .sku-ref img { width: 128px; height: 128px; object-fit: contain; border-radius: 6px;
                 border: 1px solid #eee; background: #fafafa; cursor: zoom-in; }
  .sku-ref-placeholder { width: 128px; height: 128px; border-radius: 6px; background: #eee;
                         border: 1px solid #ddd; flex-shrink: 0; }
  .sku-ref .meta { font-size: 0.75rem; line-height: 1.2; }
  .sku-ref .meta .id { font-weight: 600; color: #333; }
  .sku-ref .meta .label { color: #888; }
  .sku-ref .meta .count { color: #aaa; font-size: 0.7rem; }
  .sidebar .empty { font-size: 0.8rem; color: #aaa; padding: 12px 8px; text-align: center; }
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .toolbar { display: flex; align-items: center; gap: 10px; padding: 10px 20px;
             background: #f5f5f5; border-bottom: 1px solid #ddd; flex-shrink: 0; }
  .toolbar h1 { font-size: 1.2rem; }
  .toolbar .info { font-size: 0.85rem; color: #888; }
  #done-btn { margin-left: auto; padding: 8px 20px; background: #2563eb; color: #fff;
              border: none; border-radius: 8px; font-size: 0.95rem; font-weight: 600;
              cursor: pointer; }
  #done-btn:hover { background: #1d4ed8; }
  #done-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  #skip-all-btn { padding: 8px 16px; background: #6b7280; color: #fff; border: none;
                  border-radius: 8px; font-size: 0.85rem; cursor: pointer; }
  #skip-all-btn:hover { background: #4b5563; }
  #done-msg { display: none; padding: 40px; text-align: center; }
  #done-msg h1 { font-size: 1.5rem; margin-bottom: 8px; }
  #done-msg p { color: #666; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
          gap: 12px; padding: 16px 20px; overflow-y: auto; flex: 1; }
  .card { background: #fff; border-radius: 10px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);
          border-left: 4px solid var(--zone-color, #888); }
  .card.high { --zone-color: #22c55e; }
  .card.ambiguous { --zone-color: #eab308; }
  .card.none { --zone-color: #ef4444; }
  .card.confirmed { opacity: 0.75; }
  .card.skipped { opacity: 0.5; }
  .card-header { display: flex; justify-content: space-between; align-items: center; }
  .card-header-left { display: flex; align-items: center; gap: 6px; }
  .status-badge { display: inline-block; font-size: 0.7rem; padding: 2px 10px; border-radius: 10px;
                  font-weight: 600; }
  .badge-pending { background: #fef9c3; color: #854d0e; }
  .badge-confirmed { background: #dcfce7; color: #166534; }
  .badge-skipped { background: #f3f4f6; color: #6b7280; }
  .image-pair { display: flex; gap: 10px; margin: 8px 0; }
  .image-pair figure { flex: 1; text-align: center; }
  .image-pair figcaption { font-size: 0.75rem; color: #888; margin-top: 2px; }
  .image-pair img { width: 100%; height: 140px; object-fit: contain; background: #fafafa;
                    border-radius: 6px; border: 1px solid #eee; }
  .info-row { font-size: 0.85rem; margin: 4px 0; display: flex; gap: 12px; flex-wrap: wrap; }
  .info-row .sku-label { font-weight: 600; }
  .info-row .score { color: #555; }
  .info-row .comp-scores { font-size: 0.75rem; color: #999; }
  .actions { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
  .actions button { padding: 5px 12px; border: none; border-radius: 6px; cursor: pointer;
                    font-size: 0.8rem; font-weight: 500; }
  .btn-accept { background: #22c55e; color: #fff; }
  .btn-accept:hover { background: #16a34a; }
  .btn-rename { background: #eab308; color: #fff; }
  .btn-rename:hover { background: #ca8a04; }
.btn-create { background: #ef4444; color: #fff; }
.btn-create:hover { background: #dc2626; }
.btn-quick { background: #8b5cf6; color: #fff; padding: 6px 10px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.75rem; white-space: nowrap; }
.btn-quick:hover { background: #7c3aed; }
  .btn-skip { background: #9ca3af; color: #fff; }
  .btn-skip:hover { background: #6b7280; }
  .btn-ambiguous-sku { background: #f97316; color: #fff; }
  .btn-ambiguous-sku:hover { background: #ea580c; }
  .actions button:disabled { opacity: 0.35; cursor: not-allowed; }
  .field { display: flex; gap: 6px; align-items: center; margin-top: 4px; flex-wrap: wrap; }
  .field input, .field select { padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px;
                                 font-size: 0.8rem; }
  .hide { display: none; }
  .lightbox { position: fixed; inset: 0; z-index: 9999; display: none;
              align-items: center; justify-content: center;
              background: rgba(0,0,0,0); cursor: zoom-out; }
  .lightbox.open { display: flex; }
  .lightbox-bg { position: absolute; inset: 0; background: rgba(0,0,0,0.7);
                 opacity: 0; transition: opacity 0.12s ease; }
  .lightbox.open .lightbox-bg { opacity: 1; }
  .lightbox img { max-width: 92vw; max-height: 92vh; object-fit: contain;
                  border-radius: 8px; box-shadow: 0 8px 40px rgba(0,0,0,0.5);
                  transform: scale(0.85); transition: transform 0.15s ease;
                  cursor: zoom-out; }
  .lightbox.open img { transform: scale(1); }
  /* SKU manual selector */
  .sku-selector-row { display: flex; gap: 6px; align-items: center; margin-top: 6px; flex-wrap: wrap; }
  .sku-selector-row select { flex: 1; min-width: 160px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.8rem; }
  .sku-selector-row input { flex: 0 0 130px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.8rem; }
  .btn-assign { background: #2563eb; color: #fff; padding: 5px 12px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-weight: 500; }
  .btn-assign:hover { background: #1d4ed8; }
  .undo-btn { background: #f0ad4e; color: #fff; border: none; padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; margin-left: 6px; transition: background 0.15s; vertical-align: middle; }
  .undo-btn:hover { background: #ec971f; }
  .undo-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  /* Button loading spinner — shows processing feedback after click */
  @keyframes btn-spin { to { transform: rotate(360deg); } }
  .btn-loading {
    position: relative; color: transparent !important; pointer-events: none;
  }
  .btn-loading::after {
    content: ""; position: absolute; inset: 0; margin: auto;
    width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff; border-radius: 50%;
    animation: btn-spin 0.6s linear infinite;
  }
</style>
</head>
<body>
<div class="sidebar" id="sidebar"></div>
<div class="main" id="main-area">
  <div class="toolbar">
    <h1>SKU Review: {{ title }}</h1>
    <span class="info" id="registry-info">Registry: 0 SKUs &middot; Next: SKU-001</span>
    <span class="info" id="progress">0 / 0</span>
    <button id="skip-all-btn" onclick="skipAllRemaining()">Skip All Remaining</button>
    <button id="done-btn" onclick="onDone()">Done</button>
  </div>
  <div class="grid" id="review-grid"></div>
  <div id="done-msg">
    <h1>Saved!</h1>
    <p id="done-summary"></p>
  </div>
</div>
<div class="lightbox" id="lightbox" onclick="closeLightbox(event)">
  <div class="lightbox-bg"></div>
  <img id="lightbox-img" src="" alt="enlarged crop">
</div>

<script>
const ITEMS = {{ items_json }};
const TOTAL = ITEMS.length;
const SKUS = {{ skus_json }};
let NEXT_SKU = "{{ next_sku }}";
const CATS = {{ cat_options_json }};
const DEFAULT_CAT = CATS[0] || "Beverages";

let cropStatus = {};
let cropAction = {};
for (let i = 0; i < TOTAL; i++) { cropStatus[i] = "pending"; cropAction[i] = null; }

function qs(s) { return document.querySelector(s); }

function renderSidebar() {
  const sb = document.getElementById('sidebar');
  let html = '<h2>SKU References</h2>';
  if (SKUS.length === 0) { html += '<div class="empty">No SKUs yet.<br>Create one to see it here.</div>'; }
  else {
    for (const sku of SKUS) {
      const imgUrl = sku.anchor_exemplar ? '/exemplar/' + encodeURIComponent(sku.anchor_exemplar) : '';
      html += '<div class="sku-ref" onclick="scrollToSku(\'' + sku.sku_id + '\')">';
      if (imgUrl) { html += '<img src="' + imgUrl + '" alt="" onclick="event.stopPropagation();openLightbox(this.src)" onerror="this.onerror=null;this.style.display=\'none\'">'; }
      else { html += '<div class="sku-ref-placeholder"></div>'; }
      html += '<div class="meta"><div class="id">' + sku.sku_id + '</div>';
      html += '<div class="label">' + escapeHtml(sku.label) + '</div>';
      html += '<div class="count">' + sku.n_exemplars + ' exemplar(s)</div></div></div>';
    }
  }
  sb.innerHTML = html;
}

function scrollToSku(skuId) {
  const cards = document.querySelectorAll('.card');
  for (const card of cards) {
    if (card.querySelector('.sku-label')?.textContent?.trim() === skuId) {
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card.style.boxShadow = '0 0 0 3px #2563eb';
      setTimeout(() => { card.style.boxShadow = ''; }, 1500);
      break;
    }
  }
}

function escapeHtml(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function skuOptions(filter) {
  const f = (filter || '').toLowerCase();
  let html = '';
  for (const sku of SKUS) {
    if (!f || sku.sku_id.toLowerCase().includes(f) || sku.label.toLowerCase().includes(f)) {
      html += '<option value="' + sku.sku_id + '">' + sku.sku_id + ' \u2014 ' + escapeHtml(sku.label) + '</option>';
    }
  }
  return html;
}

function filterSkuDropdown(inputId, selectId) {
  const input = document.getElementById(inputId);
  const select = document.getElementById(selectId);
  if (!input || !select) return;
  select.innerHTML = skuOptions(input.value);
}

function assignSku(idx) {
  const select = document.getElementById('sku-select-' + idx);
  const input = document.getElementById('sku-input-' + idx);
  const typed = input ? input.value.trim() : '';

  // 1. Dropdown selection → assign to existing SKU
  if (select.value) {
    commitSingle(idx, 'accept', select.value);
    return;
  }

  // 2. Typed text → match existing or create new
  if (typed) {
    // Check for exact match by SKU ID or label
    const match = SKUS.find(s =>
      s.sku_id.toLowerCase() === typed.toLowerCase() ||
      s.label.toLowerCase() === typed.toLowerCase()
    );
    if (match) {
      commitSingle(idx, 'accept', match.sku_id);
    } else {
      // No match → create new SKU with typed text as label
      if (!cropStatus[idx] || cropStatus[idx] !== "pending") return;
      doCommit({ idx: idx, action: "create", label: typed, category: DEFAULT_CAT }, idx);
    }
    return;
  }

  alert('Select or type a SKU ID or label');
}

function renderCard(i) {
  const item = ITEMS[i];
  const status = cropStatus[i] || "pending";
  const zone = item.zone || "none";
  const score = item.score != null ? item.score.toFixed(4) : "\u2014";
  const dino = item.dino_score != null ? item.dino_score.toFixed(4) : "\u2014";
  const mnet = item.mnet_score != null ? item.mnet_score.toFixed(4) : "\u2014";
  const skuId = item.sku_id || "";
  const label = item.label || "";
  const cropFile = (item.crop_path || "").split('/').pop();
  const anchorUrl = skuId ? '/anchors/' + skuId + '/crop_000.jpg' : '';
  const candidates = item.candidates || [];

  const confirmed = status === "confirmed";
  const skipped = status === "skipped";
  const pending = status === "pending";
  const noMatch = skuId === null || skuId === "";

  let badgeClass = "badge-pending";
  let badgeText = "Pending";
  if (confirmed) { badgeClass = "badge-confirmed"; badgeText = "Confirmed"; }
  if (skipped) { badgeClass = "badge-skipped"; badgeText = "Skipped"; }

  let actionHtml = "";
  let extraHtml = "";

  let undoHtml = "";
  if (confirmed) {
    undoHtml = ' <button class="undo-btn" data-idx="' + i + '" onclick="this.classList.add(\'btn-loading\');undoCrop(' + i + ')" title="Revert to pending">\u21a9 Undo</button>';
  }

    if (pending) {
      if (zone === "high" && !noMatch) {
        actionHtml += '<button class="btn-accept" data-idx="' + i + '" onclick="this.classList.add(\'btn-loading\');commitSingle(' + i + ',\'accept\')">Accept</button>';
        actionHtml += '<button class="btn-rename" data-idx="' + i + '" onclick="toggleRename(' + i + ')">Rename</button>';
      extraHtml += '<div class="field hide" id="rename-' + i + '">';
      extraHtml += '<input id="label-' + i + '" value="' + escapeHtml(label) + '">';
      extraHtml += '<select id="cat-' + i + '">' + catOpts() + '</select>';
      extraHtml += '<button data-idx="' + i + '" onclick="this.classList.add(\'btn-loading\');commitRename(' + i + ')">Save</button></div>';
    }

    if (zone === "ambiguous" && candidates.length > 0) {
      for (const c of candidates) {
        actionHtml += '<button class="btn-ambiguous-sku" data-idx="' + i + '" onclick="this.classList.add(\'btn-loading\');commitSingle(' + i + ',\'accept\',\'' + c.sku_id + '\')">' + c.sku_id + ' (' + c.score.toFixed(2) + ')</button>';
      }
    }

    // Manual SKU selector for all pending cards
    extraHtml += '<div class="sku-selector-row">';
    extraHtml += '<select id="sku-select-' + i + '">' + skuOptions() + '</select>';
    extraHtml += '<input id="sku-input-' + i + '" placeholder="(or type SKU ID)" oninput="filterSkuDropdown(\'sku-input-\' + ' + i + ', \'sku-select-\' + ' + i + ')">';
    extraHtml += '<button class="btn-assign" data-idx="' + i + '" onclick="this.classList.add(\'btn-loading\');assignSku(' + i + ')">Assign</button>';
    extraHtml += '</div>';

    actionHtml += '<button class="btn-skip" data-idx="' + i + '" onclick="this.classList.add(\'btn-loading\');commitSingle(' + i + ',\'skip\')">Skip</button>';
    if ((zone === "ambiguous" && candidates.length > 0) || noMatch || zone === "none") {
      actionHtml += '<button class="btn-quick" data-idx="' + i + '" onclick="this.classList.add(\'btn-loading\');commitCreateQuick(' + i + ')">+ ' + NEXT_SKU + '</button>';
    }
  }

  const extraClass = confirmed ? " confirmed" : (skipped ? " skipped" : "");
  const cardClass = "card " + zone + extraClass;

  return '<div class="' + cardClass + '" id="card-' + i + '" data-sku="' + escapeHtml(skuId) + '">' +
    '<div class="card-header"><div class="card-header-left">' +
    '<input type="checkbox" class="bulk-cb" data-idx="' + i + '" title="Select" ' + (!pending ? 'disabled' : '') + '>' +
    '<span class="info-row"><strong>Crop #' + i + '</strong></span></div>' +
    '<span class="status-badge ' + badgeClass + '">' + badgeText + undoHtml + '</span></div>' +
    '<div class="image-pair"><figure>' +
    '<img src="/crops/' + cropFile + '" alt="Crop ' + i + '"><figcaption>Crop</figcaption></figure>' +
    '<figure>' +
    (anchorUrl ? '<img src="' + anchorUrl + '" alt="Anchor" onerror="this.parentElement.style.display=\'none\'">' : '<div class="no-anchor">No anchor</div>') +
    '<figcaption>Anchor: ' + (skuId || "\u2014") + '</figcaption></figure></div>' +
    '<div class="info-row"><span class="sku-label">' + (skuId || "NEW") + '</span>' +
    '<span class="score"> &mdash; ' + escapeHtml(label) + ' &middot; <strong>' + score + '</strong></span>' +
    '<span class="comp-scores">DINO: ' + dino + ' &middot; MNet: ' + mnet + '</span></div>' +
    '<div class="actions">' + actionHtml + '</div>' +
    extraHtml +
    '</div>';
}

function renderGrid() {
  let html = "";
  for (let i = 0; i < TOTAL; i++) { html += renderCard(i); }
  document.getElementById('review-grid').innerHTML = html;
  updateToolbar();
}

function catOpts() {
  const cats = {{ cat_options_json }};
  return cats.map(c => '<option value="' + c + '">' + c + '</option>').join('');
}

function updateToolbar() {
  let pending = 0, confirmed = 0, skipped = 0;
  for (let i = 0; i < TOTAL; i++) {
    const s = cropStatus[i];
    if (s === "confirmed") confirmed++;
    else if (s === "skipped") skipped++;
    else pending++;
  }
  document.getElementById('progress').textContent = confirmed + " confirmed, " + skipped + " skipped, " + pending + " pending";
  const doneBtn = document.getElementById('done-btn');
  doneBtn.disabled = pending > 0;
  doneBtn.textContent = pending > 0 ? "Done (" + pending + " pending)" : "Done";
}

function commitSingle(idx, action, skuId) {
  if (cropStatus[idx] !== "pending") return;
  const body = { idx: idx, action: action };
  if (action === "accept") body.sku_id = skuId || ITEMS[idx].sku_id;
  doCommit(body, idx);
}

function undoCrop(idx) {
  if (cropStatus[idx] !== "confirmed") return;
  const body = { idx: idx, action: "reject" };
  doCommit(body, idx);
}

function commitCreateQuick(idx) {
  if (cropStatus[idx] !== "pending") return;
  doCommit({ idx: idx, action: "create", label: NEXT_SKU, category: DEFAULT_CAT }, idx);
}

function toggleRename(idx) {
  const el = document.getElementById('rename-' + idx);
  if (el) el.classList.toggle('hide');
}
function commitRename(idx) {
  if (cropStatus[idx] !== "pending") return;
  const label = document.getElementById('label-' + idx)?.value?.trim();
  if (!label) { alert('Enter a label'); return; }
  const cat = document.getElementById('cat-' + idx)?.value || "Other";
  doCommit({ idx: idx, action: "rename", sku_id: ITEMS[idx].sku_id, label: label, category: cat }, idx);
}

function doCommit(body, singleIdx) {
  const bulkCbs = document.querySelectorAll('.bulk-cb:checked');
  const bulkIdxs = [];
  bulkCbs.forEach(cb => {
    const ci = parseInt(cb.getAttribute('data-idx'));
    if (ci !== singleIdx && cropStatus[ci] === "pending") bulkIdxs.push(ci);
  });
  const allIdxs = [singleIdx];
  for (const bi of bulkIdxs) { if (!allIdxs.includes(bi)) allIdxs.push(bi); }

  body.indices = allIdxs;

  // Track which card's buttons to unload on response
  const loadingBtns = document.querySelectorAll('[data-idx="' + singleIdx + '"]');

  fetch('/api/commit', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })
  .then(r => r.json())
  .then(data => {
    loadingBtns.forEach(function(b) { b.classList.remove('btn-loading'); });
    if (data.error) { alert(data.error); return; }
    for (const entry of data.committed) {
      if (entry.action === 'reject') {
        cropStatus[entry.idx] = "pending";
        delete cropAction[entry.idx];
      } else {
        cropStatus[entry.idx] = entry.status || "confirmed";
        cropAction[entry.idx] = entry;
      }
    }
    if (data.rescore) {
      for (const [idxStr, update] of Object.entries(data.rescore)) {
        const idx = parseInt(idxStr);
        if (cropStatus[idx] === "pending") {
          Object.assign(ITEMS[idx], update);
        }
      }
    }
    if (data.skus) {
      SKUS.length = 0;
      SKUS.push(...data.skus);
    }
    if (data.registry_count != null) {
      document.getElementById('registry-info').textContent =
        "Registry: " + data.registry_count + " SKUs &middot; Next: " + (data.next_sku || "\u2014");
      if (data.next_sku) NEXT_SKU = data.next_sku;
    }
    if (data.log) { console.log("[server]", data.log); }
    renderGrid();
    renderSidebar();
  })
  .catch(err => {
    loadingBtns.forEach(function(b) { b.classList.remove('btn-loading'); });
    console.error("Commit failed:", err);
    alert("Commit failed \u2014 see console");
  });
}

function skipAllRemaining() {
  const pending = [];
  for (let i = 0; i < TOTAL; i++) {
    if (cropStatus[i] === "pending") pending.push(i);
  }
  if (pending.length === 0) return;
  if (!confirm("Skip " + pending.length + " remaining pending crops?")) return;

  fetch('/api/commit', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: "batch-skip", indices: pending })
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { alert(data.error); return; }
    for (const entry of data.committed) {
      cropStatus[entry.idx] = "skipped";
      cropAction[entry.idx] = entry;
    }
    renderGrid();
  });
}

function onDone() {
  const pending = [];
  for (let i = 0; i < TOTAL; i++) {
    if (cropStatus[i] === "pending") pending.push(i);
  }
  if (pending.length > 0) {
    alert(pending.length + " crops still pending \u2014 confirm or skip them first.");
    return;
  }

  const doneBtn = document.getElementById('done-btn');
  doneBtn.disabled = true;
  doneBtn.textContent = "Saving...";

  fetch('/api/done', { method: 'POST' })
  .then(r => r.json())
  .then(data => {
    if (data.error) { doneBtn.disabled = false; doneBtn.textContent = "Done"; alert(data.error); return; }
    const nConfirmed = data.summary?.confirmed || 0;
    const nSkipped = data.summary?.skipped || 0;
    document.getElementById('main-area').querySelector('.toolbar')?.remove();
    document.getElementById('review-grid')?.remove();
    const msg = document.getElementById('done-msg');
    msg.style.display = 'block';
    document.getElementById('done-summary').textContent =
      nConfirmed + " crops confirmed, " + nSkipped + " skipped. You can close this tab.";
  })
  .catch(err => { doneBtn.disabled = false; doneBtn.textContent = "Done"; alert("Failed: " + err.message); });
}

function openLightbox(src) {
  const lb = document.getElementById('lightbox');
  document.getElementById('lightbox-img').src = src;
  lb.classList.add('open');
}
function closeLightbox(e) {
  if (e.target === e.currentTarget || e.target.classList.contains('lightbox-bg')) {
    document.getElementById('lightbox').classList.remove('open');
  }
}
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('open');
});
// Wire crop images to open lightbox (delegated since grid rerenders)
document.getElementById('review-grid').addEventListener('click', function(e) {
  const img = e.target.closest('.image-pair img');
  if (img) openLightbox(img.src);
});

renderSidebar();
renderGrid();
</script>
</body>
</html>
"""


class _ReviewHandler(BaseHTTPRequestHandler):
    session: ReviewSession

    def log_message(self, fmt, *args) -> None:
        logger.debug(fmt, *args)

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _serve_static(self, base_dir: Path) -> None:
        rel = self.path.split("/", 2)[-1]
        fpath = base_dir / rel
        if not fpath.exists() or not fpath.is_file():
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        ext = fpath.suffix.lower()
        ct_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".png": "image/png", ".webp": "image/webp"}
        self.send_header("Content-Type", ct_map.get(ext, "application/octet-stream"))
        self.end_headers()
        self.wfile.write(fpath.read_bytes())

    # ── Routes ──────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._serve_review_page()
        elif self.path.startswith("/crops/"):
            self._serve_static(self.session.crops_dir)
        elif self.path.startswith("/anchors/"):
            self._serve_static(config.PIPELINE_DIR / "exemplars")
        elif self.path.startswith("/exemplar/"):
            self._serve_exemplar()
        elif self.path == "/api/status":
            self._send_json(self._build_status())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/commit":
            self._handle_commit()
        elif self.path == "/api/done":
            self._handle_done()
        else:
            self.send_response(404)
            self.end_headers()

    # ── Exemplar serving (from registry paths) ──────────────────────────────

    def _serve_exemplar(self) -> None:
        import urllib.parse
        rel = urllib.parse.unquote(self.path.split("/", 2)[-1])
        fpath = Path(rel)
        if rel and fpath.exists() and fpath.is_file():
            self._respond_with_image(fpath)
            return

        # Fallback: registry paths may be stale (e.g. workspace was flushed).
        # Search for the filename in the exemplars/ directory.
        fname = Path(rel).name
        if fname:
            ex_dir = config.PIPELINE_DIR / "exemplars"
            if ex_dir.is_dir():
                for f in ex_dir.rglob(fname):
                    if f.is_file():
                        self._respond_with_image(f)
                        return

        self.send_response(404)
        self.end_headers()

    def _respond_with_image(self, fpath: Path) -> None:
        ext = fpath.suffix.lower()
        ct = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "application/octet-stream"
        body = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── Review page ─────────────────────────────────────────────────────────

    def _serve_review_page(self) -> None:
        html = _TEMPLATE
        html = html.replace("{{ title }}", self.session.title)
        html = html.replace("{{ items_json }}", json.dumps(self.session.items))
        html = html.replace("{{ skus_json }}", json.dumps(self.session.sku_list))
        html = html.replace("{{ cat_options_json }}", json.dumps(CATEGORIES))
        html = html.replace("{{ next_sku }}", self.session.next_sku)

        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    # ── Commit ──────────────────────────────────────────────────────────────

    def _handle_commit(self) -> None:
        body = self._read_body()
        action = body.get("action")
        indices = body.get("indices", [])
        idx = body.get("idx")

        if not indices and idx is not None:
            indices = [idx]

        if not indices:
            self._send_error("No indices provided")
            return

        session = self.session
        committed: list[dict] = []

        try:
            if action == "accept":
                sku_id = body.get("sku_id") or session.items[indices[0]].get("sku_id")
                if not sku_id:
                    self._send_error("No sku_id for accept")
                    return
                results = session.batch_commit(indices, "accept", sku_id=sku_id)
                committed = [
                    {"idx": r["idx"], "sku_id": r["sku_id"],
                     "status": "confirmed", "action": "accept"}
                    for r in results
                ]

            elif action == "create":
                label = body.get("label", "")
                category = body.get("category", "Other")
                results = session.batch_commit(
                    indices, "create", label=label, category=category,
                )
                committed = [
                    {"idx": r["idx"], "sku_id": r["sku_id"],
                     "status": "confirmed", "action": "create"}
                    for r in results
                ]

            elif action == "rename":
                sku_id = body.get("sku_id") or session.items[indices[0]].get("sku_id")
                label = body.get("label", "")
                category = body.get("category", "Other")
                results = session.batch_commit(
                    indices, "rename",
                    sku_id=sku_id, label=label, category=category,
                )
                committed = [
                    {"idx": r["idx"], "sku_id": r["sku_id"],
                     "status": "confirmed", "action": "rename"}
                    for r in results
                ]

            elif action in ("skip", "batch-skip"):
                session.batch_commit(indices, "skip")
                committed = [
                    {"idx": i, "status": "skipped", "action": "skip"}
                    for i in indices
                ]

            elif action == "reject":
                results = session.batch_commit(indices, "reject")
                committed = [
                    {"idx": r["idx"], "status": "pending", "action": "reject"}
                    for r in results
                ]

            else:
                self._send_error(f"Unknown action: {action}")
                return

        except ValueError as e:
            self._send_error(str(e))
            return

        # Sync exemplar + master_reference for newly created SKUs.
        # Existing SKU anchors already exist; only new SKUs need syncing.
        # Runs in a background thread so the UI response isn't delayed.
        if action == "create":
            def _sync_new_sku(sku_id: str, reg: SKURegistry) -> None:
                reg.sync_exemplar_files(sku_id)
                # Also copy the first exemplar to master_references/
                src = config.PIPELINE_DIR / "exemplars" / sku_id / "crop_000.jpg"
                if src.exists():
                    dst = config.PIPELINE_DIR / "master_references" / f"{sku_id}.jpg"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst))
            for r in committed:
                sku = r.get("sku_id")
                if sku:
                    t = threading.Thread(
                        target=_sync_new_sku,
                        args=(sku, session.registry),
                        daemon=True,
                    )
                    t.start()

        rescore = session.rescore_pending()

        response = {
            "status": "ok",
            "committed": committed,
            "rescore": {str(k): v for k, v in rescore.items()},
            "skus": session.sku_list,
            "registry_count": len(session.registry),
            "next_sku": session.next_sku,
            "log": f"Committed {action} on indices {indices}",
        }
        self._send_json(response)

    # ── Done ────────────────────────────────────────────────────────────────

    def _handle_done(self) -> None:
        try:
            outcomes = self.session.finalize()
        except ValueError as e:
            self._send_error(str(e))
            return

        type(self).done_called = True
        self.session.print_outcome_table(outcomes)

        self._send_json({
            "status": "ok",
            "summary": {
                "confirmed": len(self.session.confirmed_indices),
                "skipped": len(self.session.skipped_indices),
                "total": len(self.session.items),
            },
        })

    # ── Status ──────────────────────────────────────────────────────────────

    def _build_status(self) -> dict:
        return {
            "total": len(self.session.items),
            "pending": self.session.pending_indices,
            "confirmed": self.session.confirmed_indices,
            "skipped": self.session.skipped_indices,
            "registry_count": len(self.session.registry),
            "next_sku": self.session.next_sku,
            "skus": self.session.sku_list,
        }


# ── Server launcher ──────────────────────────────────────────────────────────


def launch_server(session: ReviewSession) -> dict[int, dict]:
    """Launch the review server, open browser, block until confirmed.

    Returns the ``{idx: outcome}`` dict after Done is pressed.
    """
    _ReviewHandler.session = session

    server = HTTPServer((config.REVIEW_HOST, config.REVIEW_PORT), _ReviewHandler)
    url = f"http://{config.REVIEW_HOST}:{config.REVIEW_PORT}"

    logger.info("Opening review page at %s", url)
    webbrowser.open(url)

    _ReviewHandler.done_called = False

    try:
        server.handle_request()
        while not _ReviewHandler.done_called:
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    confirmed_actions: dict[int, dict] = {}
    for idx in range(len(session.items)):
        action = session.crop_action.get(idx, {})
        if action.get("action") in ("accept", "create", "new", "rename"):
            confirmed_actions[idx] = action

    # Guarantee the in-memory registry is flushed to disk before returning.
    # This ensures step_export and subsequent runs see the correct data.
    session.registry._save()

    logger.info(
        "Review complete: %d items confirmed", len(confirmed_actions),
    )
    return confirmed_actions
