import React, { useEffect, useRef } from 'react';
import { Play, Pause } from 'lucide-react';
import type { ConversationEntry } from '../types';
import { renderMarkdown } from '../markdown';
import styles from '../firecastbot.module.scss';

type Props = {
  conversation: ConversationEntry[];
  isQuerying: boolean;
  isSpeaking: boolean;
  isSpeechPaused: boolean;
  activeSpeechMessageKey: string;
  startBrowserSpeech: (text: string, key: string) => void;
  resumeBrowserSpeech: (key: string) => void;
  pauseBrowserSpeech: (key: string) => void;
};

export function ConversationPanel({
  conversation,
  isQuerying,
  isSpeaking,
  isSpeechPaused,
  activeSpeechMessageKey,
  startBrowserSpeech,
  resumeBrowserSpeech,
  pauseBrowserSpeech,
}: Props) {
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation, isQuerying]);

  return (
    <div className={styles.panel}>
      <h2>Conversation</h2>
      <div className={styles.messages}>
        {conversation.length === 0 && !isQuerying && (
          <p className={styles.empty}>No messages yet.</p>
        )}
        {conversation.map((entry, index) => {
          const messageKey = `${entry.role}-${index}`;
          const isThisMessagePaused = activeSpeechMessageKey === messageKey && isSpeechPaused;
          return (
            <div
              key={messageKey}
              className={entry.role === 'assistant' ? styles.assistantMessage : styles.userMessage}
            >
              <div className={styles.messageRole}>{entry.role}</div>
              <div className={styles.messageBody}>
                {entry.role === 'assistant' ? renderMarkdown(entry.content) : entry.content}
              </div>
              {entry.role === 'assistant' && (
                <div className={styles.messageControls}>
                  <button
                    type="button"
                    className={`${styles.messageControlButton} ${styles.playButton}`}
                    onClick={() => {
                      if (activeSpeechMessageKey === messageKey && isSpeaking && isSpeechPaused) {
                        resumeBrowserSpeech(messageKey);
                      } else {
                        startBrowserSpeech(entry.content, messageKey);
                      }
                    }}
                    aria-label={isThisMessagePaused ? 'Resume' : 'Play'}
                    data-tooltip={isThisMessagePaused ? 'Resume' : 'Play'}
                  >
                    <Play size={14} />
                  </button>
                  <button
                    type="button"
                    className={`${styles.messageControlButton} ${styles.pauseButton}`}
                    onClick={() => pauseBrowserSpeech(messageKey)}
                    disabled={
                      activeSpeechMessageKey !== messageKey || !isSpeaking || isSpeechPaused
                    }
                    aria-label="Pause"
                    data-tooltip="Pause"
                  >
                    <Pause size={14} />
                  </button>
                </div>
              )}
            </div>
          );
        })}
        {isQuerying && (
          <div className={styles.typingIndicator}>
            <span />
            <span />
            <span />
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}
