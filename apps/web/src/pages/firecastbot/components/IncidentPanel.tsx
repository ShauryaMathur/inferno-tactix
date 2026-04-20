import React from "react";
import { API_BASE_URL } from "../../../env";
import type { ConversationEntry, Preset } from "../types";
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
  conversation: ConversationEntry[];
  sessionReadyAt: Date | null;
};

function exportConversationAsPdf(
  conversation: ConversationEntry[],
  incidentName: string,
  sessionReadyAt: Date | null,
) {
  const dateStr = sessionReadyAt
    ? sessionReadyAt.toLocaleString([], { dateStyle: "medium", timeStyle: "short" })
    : new Date().toLocaleString([], { dateStyle: "medium", timeStyle: "short" });

  const rows = conversation
    .map((entry) => {
      const role = entry.role === "user" ? "You" : "FireCastBot";
      const roleClass = entry.role === "user" ? "user" : "assistant";
      const text = entry.content
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\n/g, "<br>");
      return `<div class="message ${roleClass}"><span class="role">${role}</span><p>${text}</p></div>`;
    })
    .join("");

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FireCastBot — ${incidentName}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; font-size: 13px; color: #192b1c; padding: 32px 40px; max-width: 820px; margin: 0 auto; }
  header { border-bottom: 2px solid #2e7d52; padding-bottom: 14px; margin-bottom: 24px; }
  header h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.02em; }
  header p { margin-top: 4px; font-size: 11px; color: #5a7a62; }
  .message { margin-bottom: 16px; }
  .role { display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 4px; }
  .user .role { color: #1a5c36; }
  .assistant .role { color: #4a6e5a; }
  .message p { line-height: 1.6; color: #192b1c; padding-left: 2px; }
  .user p { font-weight: 500; }
  @media print { body { padding: 20px 24px; } }
</style>
</head>
<body>
<header>
  <h1>FireCastBot — ${incidentName}</h1>
  <p>Session started ${dateStr} &nbsp;·&nbsp; ${conversation.length} message${conversation.length !== 1 ? "s" : ""}</p>
</header>
${rows}
</body>
</html>`;

  const win = window.open("", "_blank");
  if (!win) return;
  win.document.write(html);
  win.document.close();
  win.focus();
  setTimeout(() => win.print(), 250);
}

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
  conversation,
  sessionReadyAt,
}: Props) {
  const incidentName = selectedPdfName || "Conversation";
  const canExport = conversation.length > 0;

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

      <div className={styles.exportSection}>
        <p className={styles.fieldLabel}>Export</p>
        <button
          type="button"
          className={styles.exportButton}
          disabled={!canExport}
          title={canExport ? "Export conversation as PDF" : "No conversation to export yet"}
          onClick={() => exportConversationAsPdf(conversation, incidentName, sessionReadyAt)}
        >
          Export as PDF
        </button>
      </div>
    </div>
  );
}
