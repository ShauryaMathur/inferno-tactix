import React from 'react';
import { Settings } from 'lucide-react';
import type { Provider, FireCastBotConfig } from '../types';
import styles from '../firecastbot.module.scss';

type Props = {
  isBotReady: boolean;
  isLoading: boolean;
  error: string;
  retryBootstrap: () => void;
  sessionReadyAt: Date | null;
  showSpeechSettings: boolean;
  setShowSpeechSettings: (v: boolean | ((prev: boolean) => boolean)) => void;
  settingsPanelRef: React.RefObject<HTMLDivElement | null>;
  providers: FireCastBotConfig['providers'];
  speechToTextProviderId: string;
  setSpeechToTextProviderId: (v: string) => void;
  selectedSttProvider: Provider | null;
  speakResponses: boolean;
  setSpeakResponses: (v: boolean) => void;
};

export function HeroSection({
  isBotReady,
  isLoading,
  error,
  retryBootstrap,
  sessionReadyAt,
  showSpeechSettings,
  setShowSpeechSettings,
  settingsPanelRef,
  providers,
  speechToTextProviderId,
  setSpeechToTextProviderId,
  selectedSttProvider,
  speakResponses,
  setSpeakResponses,
}: Props) {
  return (
    <section className={styles.hero}>
      <div className={styles.heroSettings}>
        <div className={styles.settingsPanel} ref={settingsPanelRef}>
          <button
            type="button"
            className={styles.settingsButton}
            onClick={() => setShowSpeechSettings((open) => !open)}
            aria-expanded={showSpeechSettings}
            aria-label="Toggle speech settings"
          >
            <Settings size={16} />
          </button>

          {showSpeechSettings && (
            <div className={styles.settingsDropdown}>
              <label className={styles.fieldLabel}>Speech to text</label>
              <select
                value={speechToTextProviderId}
                onChange={(e) => setSpeechToTextProviderId(e.target.value)}
              >
                {providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>

              {selectedSttProvider?.transcriptionUnavailableReason &&
                !selectedSttProvider.transcriptionAvailable && (
                  <p className={styles.note}>
                    {selectedSttProvider.transcriptionUnavailableReason}
                  </p>
                )}

              <label className={styles.checkbox}>
                <input
                  type="checkbox"
                  checked={speakResponses}
                  onChange={(e) => setSpeakResponses(e.target.checked)}
                />
                Read responses aloud (browser)
              </label>
            </div>
          )}
        </div>
      </div>

      <div className={styles.badge}>
        <span className={`${styles.badgeDot} ${isBotReady ? styles.badgeDotReady : ''}`} />
        <span>FIRECASTBOT</span>
        {sessionReadyAt && (
          <span className={styles.sessionTime}>
            · {sessionReadyAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
        )}
      </div>
      <h2 className={styles.title}>Ask FireCastBot</h2>
      <p className={styles.subtitle}>
        Upload an incident report, extract structured facts, and query it alongside doctrine and
        safety references without leaving FireCastRL.
      </p>
      {isLoading && <p className={styles.loadingText}>Starting FireCastBot&hellip;</p>}
      {error && !isLoading && (
        <div className={styles.errorRow}>
          <p className={styles.error}>{error}</p>
          <button type="button" className={styles.retryButton} onClick={retryBootstrap}>
            Retry
          </button>
        </div>
      )}
    </section>
  );
}
