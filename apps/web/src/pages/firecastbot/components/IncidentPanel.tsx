import React from "react";
import { API_BASE_URL } from "../../../env";
import type { Preset } from "../types";
import styles from "../firecastbot.module.scss";

type Props = {
  presets: Preset[];
  isBusy: boolean;
  sessionId: string;
  isIncidentSourceLocked: boolean;
  selectedPdfName: string;
  pdfInputRef: React.RefObject<HTMLInputElement | null>;
  setSelectedPdfName: (v: string) => void;
  uploadPdf: () => void;
  ingestPreset: (id: string, label: string) => void;
  startNewSession: () => void;
};

export function IncidentPanel({
  presets,
  isBusy,
  sessionId,
  isIncidentSourceLocked,
  selectedPdfName,
  pdfInputRef,
  setSelectedPdfName,
  uploadPdf,
  ingestPreset,
  startNewSession,
}: Props) {
  return (
    <div className={styles.panel}>
      <h2>Incident Report</h2>
      <button
        type="button"
        className={styles.secondaryButton}
        onClick={startNewSession}
        disabled={isBusy}
      >
        Start New Session
      </button>

      <input
        ref={pdfInputRef}
        className={styles.hiddenInput}
        type="file"
        accept="application/pdf"
        disabled={isIncidentSourceLocked}
        onChange={(e) => setSelectedPdfName(e.target.files?.[0]?.name || "")}
      />
      <div className={`${styles.filePicker} ${isIncidentSourceLocked ? styles.lockedSection : ""}`}>
        <button
          type="button"
          className={styles.fileTrigger}
          disabled={isIncidentSourceLocked}
          onClick={() => pdfInputRef.current?.click()}
        >
          Upload Incident PDF
        </button>
        <div className={styles.fileName}>{selectedPdfName || "No file selected"}</div>
      </div>
      <button onClick={uploadPdf} disabled={isBusy || !sessionId || isIncidentSourceLocked}>
        Load Report
      </button>

      <div className={`${styles.presetSection} ${isIncidentSourceLocked ? styles.lockedSection : ""}`}>
        <p className={styles.fieldLabel}>Quick presets</p>
        <div className={styles.presetGrid}>
          {presets.map((preset) => (
            <div key={preset.id} className={styles.presetRow}>
              <button
                type="button"
                className={styles.presetButton}
                onClick={() => ingestPreset(preset.id, preset.label)}
                disabled={isBusy || !sessionId || !preset.available || isIncidentSourceLocked}
                title={
                  isIncidentSourceLocked
                    ? "This session already has an incident report loaded."
                    : preset.available
                      ? `Load ${preset.label}`
                      : `${preset.label} is not available in incident_reports`
                }
              >
                {preset.label}
              </button>
              {preset.previewUrl && preset.available && (
                <a
                  href={`${API_BASE_URL}${preset.previewUrl}`}
                  target="_blank"
                  rel="noreferrer"
                  className={styles.presetPreviewLink}
                  title={`Preview ${preset.label} PDF`}
                >
                  Preview
                </a>
              )}
            </div>
          ))}
        </div>
      </div>

      {isIncidentSourceLocked && (
        <p className={styles.note}>
          Incident source is locked for this chat session. Start a new session to load a different report.
        </p>
      )}
    </div>
  );
}
