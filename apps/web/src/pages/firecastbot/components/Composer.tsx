import React from 'react';
import { ArrowUp, Mic, MicOff } from 'lucide-react';
import styles from '../firecastbot.module.scss';

const SUGGESTED_QUESTIONS = [
  'How serious is this wildfire threat right now, and what parts of the area are most at risk?',
  'What should I do in the next few hours to prepare my home and family if this fire gets worse?',
  'Based on this incident report, should I stay ready to evacuate, and what warning signs should I watch for?',
];

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
  const showSuggestions = isBotReady && !queryInput;

  return (
    <div className={`${styles.panel} ${styles.stickyComposer}`}>
      <h2>Chat</h2>
      {showSuggestions && (
        <div className={styles.suggestions}>
          {SUGGESTED_QUESTIONS.map((q) => (
            <button
              key={q}
              type="button"
              className={styles.suggestionChip}
              onClick={() => setQueryInput(q)}
            >
              {q}
            </button>
          ))}
        </div>
      )}
      <div className={styles.composerInputWrap}>
        <textarea
          value={queryInput}
          onChange={(e) => setQueryInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if (canSubmitQuery) void submitQuery();
            }
          }}
          placeholder={
            isBotReady
              ? 'Ask a question about this wildfire situation...'
              : 'Load an incident report to start chatting...'
          }
          disabled={!isBotReady}
          rows={5}
        />
        <button
          type="button"
          className={`${styles.micButton} ${isListening || isRecording ? styles.micButtonActive : ''}`}
          onClick={handleMicClick}
          disabled={!isBotReady}
          aria-label={
            isRecording ? 'Stop recording' : isListening ? 'Listening...' : 'Speak your question'
          }
          title={
            isRecording ? 'Stop recording' : isListening ? 'Listening…' : 'Speak your question'
          }
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
