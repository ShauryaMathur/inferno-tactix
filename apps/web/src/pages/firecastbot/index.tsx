import React, { useEffect, useRef, useState } from "react";
import { useFireCastBotSession } from "./hooks/useFireCastBotSession";
import { useBrowserSpeech } from "./hooks/useBrowserSpeech";
import { useMicInput } from "./hooks/useMicInput";
import { HeroSection } from "./components/HeroSection";
import { IncidentPanel } from "./components/IncidentPanel";
import { ConversationPanel } from "./components/ConversationPanel";
import { Composer } from "./components/Composer";
import styles from "./firecastbot.module.scss";

export default function FireCastBot() {
  const [queryInput, setQueryInput] = useState("");
  const [speechToTextProviderId, setSpeechToTextProviderId] = useState("browser");
  const [speakResponses, setSpeakResponses] = useState(false);
  const [showSpeechSettings, setShowSpeechSettings] = useState(false);
  const pdfInputRef = useRef<HTMLInputElement | null>(null);
  const settingsPanelRef = useRef<HTMLDivElement | null>(null);

  const onSpeechError = useRef<(msg: string) => void>(() => {});

  const speech = useBrowserSpeech((msg) => onSpeechError.current(msg));

  const session = useFireCastBotSession({
    queryInput,
    setQueryInput,
    pdfInputRef,
    onNewReply: (reply, idx) => {
      if (speakResponses) speech.startBrowserSpeech(reply, `assistant-${idx}`);
    },
    cancelBrowserSpeech: speech.cancelBrowserSpeechInternal,
    closeSettings: () => setShowSpeechSettings(false),
  });

  // After both are initialized:
  onSpeechError.current = session.setError;

  const mic = useMicInput({
    speechToTextProviderId,
    isBotReady: session.isBotReady,
    isBusy: session.isBusy,
    getSelectedProvider: session.getSelectedProvider,
    onError: session.setError,
    setQueryInput,
    runTask: session.runTask,
    withSessionRetry: session.withSessionRetry,
    applySnapshot: session.applySnapshot,
  });

  // Sync STT provider when config loads
  useEffect(() => {
    if (session.config) {
      setSpeechToTextProviderId(session.config.defaultSpeechToTextProvider || "browser");
    }
  }, [session.config]);

  // Cancel speech on page unload
  useEffect(() => {
    const cancel = () => {
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    };
    window.addEventListener("beforeunload", cancel);
    window.addEventListener("pagehide", cancel);
    return () => {
      window.removeEventListener("beforeunload", cancel);
      window.removeEventListener("pagehide", cancel);
      cancel();
    };
  }, []);

  // Close settings on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (showSpeechSettings && settingsPanelRef.current && !settingsPanelRef.current.contains(e.target as Node)) {
        setShowSpeechSettings(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showSpeechSettings]);

  const selectedSttProvider = session.getSelectedProvider(speechToTextProviderId);
  const canSubmitQuery = !session.isBusy && !!queryInput.trim() && session.isBotReady;

  return (
    <div className={styles.container}>
      <HeroSection
        isBotReady={session.isBotReady}
        isLoading={session.isLoading}
        error={session.error}
        retryBootstrap={session.retryBootstrap}
        sessionReadyAt={session.sessionReadyAt}
        showSpeechSettings={showSpeechSettings}
        setShowSpeechSettings={setShowSpeechSettings}
        settingsPanelRef={settingsPanelRef}
        providers={session.config?.providers ?? []}
        speechToTextProviderId={speechToTextProviderId}
        setSpeechToTextProviderId={setSpeechToTextProviderId}
        selectedSttProvider={selectedSttProvider}
        speakResponses={speakResponses}
        setSpeakResponses={setSpeakResponses}
      />

      <div className={styles.layout}>
        <aside className={styles.sidebar}>
          <IncidentPanel
            presets={session.config?.presets ?? []}
            isBusy={session.isBusy}
            sessionId={session.sessionId}
            isIncidentSourceLocked={session.isIncidentSourceLocked}
            selectedPdfName={session.selectedPdfName}
            pdfInputRef={pdfInputRef}
            setSelectedPdfName={session.setSelectedPdfName}
            uploadPdf={session.uploadPdf}
            ingestPreset={session.ingestPreset}
            startNewSession={session.startNewSession}
            conversation={session.conversation}
            sessionReadyAt={session.sessionReadyAt}
          />
        </aside>

        <section className={styles.chatArea}>
          <ConversationPanel
            conversation={session.conversation}
            isQuerying={session.isQuerying}
            isSpeaking={speech.isSpeaking}
            isSpeechPaused={speech.isSpeechPaused}
            activeSpeechMessageKey={speech.activeSpeechMessageKey}
            startBrowserSpeech={speech.startBrowserSpeech}
            resumeBrowserSpeech={speech.resumeBrowserSpeech}
            pauseBrowserSpeech={speech.pauseBrowserSpeech}
          />
          <Composer
            queryInput={queryInput}
            setQueryInput={setQueryInput}
            isBotReady={session.isBotReady}
            canSubmitQuery={canSubmitQuery}
            isListening={mic.isListening}
            isRecording={mic.isRecording}
            handleMicClick={mic.handleMicClick}
            submitQuery={session.submitQuery}
          />
        </section>
      </div>
    </div>
  );
}
