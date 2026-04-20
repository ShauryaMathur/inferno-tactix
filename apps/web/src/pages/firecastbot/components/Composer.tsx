import React from "react";
import { ArrowUp, Mic, MicOff } from "lucide-react";
import styles from "../firecastbot.module.scss";

type Props = {
  queryInput: string;
  setQueryInput: (v: string) => void;
  isBotReady: boolean;
  canSubmitQuery: boolean;
  isListening: boolean;
  isRecording: boolean;
  handleMicClick: () => void;
  submitQuery: () => void;
};

export function Composer({
  queryInput,
  setQueryInput,
  isBotReady,
  canSubmitQuery,
  isListening,
  isRecording,
  handleMicClick,
  submitQuery,
}: Props) {
  return (
    <div className={`${styles.panel} ${styles.stickyComposer}`}>
      <h2>Chat</h2>
      <div className={styles.composerInputWrap}>
        <textarea
          value={queryInput}
          onChange={(e) => setQueryInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (canSubmitQuery) void submitQuery();
            }
          }}
          placeholder={isBotReady ? "Ask a question about this wildfire situation..." : "Load an incident report to start chatting..."}
          disabled={!isBotReady}
          rows={5}
        />
        <button
          type="button"
          className={`${styles.micButton} ${isListening || isRecording ? styles.micButtonActive : ""}`}
          onClick={handleMicClick}
          disabled={!isBotReady}
          aria-label={isRecording ? "Stop recording" : isListening ? "Listening..." : "Speak your question"}
          title={isRecording ? "Stop recording" : isListening ? "Listening…" : "Speak your question"}
        >
          {isListening || isRecording ? <MicOff size={15} /> : <Mic size={15} />}
        </button>
        <button
          type="button"
          className={styles.sendButton}
          onClick={submitQuery}
          disabled={!canSubmitQuery}
          aria-label="Submit message"
        >
          <ArrowUp size={16} />
        </button>
      </div>
    </div>
  );
}
