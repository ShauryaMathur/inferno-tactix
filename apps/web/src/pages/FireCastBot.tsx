import axios from "axios";
import React, { useEffect, useRef, useState } from "react";
import { ArrowUp, Settings } from "lucide-react";
import { API_BASE_URL } from "../env";
import styles from "./firecastbot.module.scss";

type Provider = {
  id: string;
  label: string;
  transcriptionAvailable: boolean;
  transcriptionUnavailableReason: string | null;
  inputMode: "upload" | "browser" | "none";
  synthesisAvailable: boolean;
  synthesisUnavailableReason: string | null;
  outputMode: "audio_bytes" | "browser" | "none";
};

type FireCastBotConfig = {
  defaultSpeechToTextProvider: string;
  defaultTextToSpeechProvider: string;
  providers: Provider[];
};

type ConversationEntry = {
  role: string;
  content: string;
};

type SessionSnapshot = {
  sessionId: string;
  documentsCount: number;
  conversation: ConversationEntry[];
  latestTranscript: string;
  latestQueryClassification?: string;
};

const SESSION_STORAGE_KEY = "firecastbot-session-id";

const getSpeechRecognition = () =>
  (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

const isUnknownSessionError = (exc: any) =>
  exc?.response?.status === 404 &&
  typeof exc?.response?.data?.error === "string" &&
  exc.response.data.error.includes("Unknown FireCastBot session");

const renderInlineMarkdown = (text: string): React.ReactNode[] => {
  const nodes: React.ReactNode[] = [];
  const pattern = /(\[([^\]]+)\]\(([^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }

    if (match[2] && match[3]) {
      nodes.push(
        <a
          key={`${match.index}-link`}
          href={match[3]}
          target="_blank"
          rel="noreferrer"
        >
          {match[2]}
        </a>,
      );
    } else if (match[4]) {
      nodes.push(<code key={`${match.index}-code`}>{match[4]}</code>);
    } else if (match[5]) {
      nodes.push(<strong key={`${match.index}-strong`}>{match[5]}</strong>);
    } else if (match[6]) {
      nodes.push(<em key={`${match.index}-em`}>{match[6]}</em>);
    }

    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes;
};

const renderMarkdown = (text: string) => {
  const normalized = text.replace(/\r\n/g, "\n");
  const lines = normalized.split("\n");
  const blocks: React.ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      blocks.push(
        <pre key={`code-${blocks.length}`} className={styles.markdownCodeBlock}>
          <code>{codeLines.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (headingMatch) {
      const content = renderInlineMarkdown(headingMatch[2]);
      const key = `heading-${blocks.length}`;
      if (headingMatch[1].length === 1) {
        blocks.push(<h1 key={key}>{content}</h1>);
      } else if (headingMatch[1].length === 2) {
        blocks.push(<h2 key={key}>{content}</h2>);
      } else {
        blocks.push(<h3 key={key}>{content}</h3>);
      }
      index += 1;
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      const items: React.ReactNode[] = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
        items.push(<li key={`bullet-${index}`}>{renderInlineMarkdown(lines[index].trim().replace(/^[-*]\s+/, ""))}</li>);
        index += 1;
      }
      blocks.push(<ul key={`ul-${blocks.length}`}>{items}</ul>);
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items: React.ReactNode[] = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
        items.push(<li key={`ordered-${index}`}>{renderInlineMarkdown(lines[index].trim().replace(/^\d+\.\s+/, ""))}</li>);
        index += 1;
      }
      blocks.push(<ol key={`ol-${blocks.length}`}>{items}</ol>);
      continue;
    }

    const paragraphLines: string[] = [];
    while (index < lines.length) {
      const current = lines[index].trim();
      if (!current || current.startsWith("```") || /^(#{1,3})\s+/.test(current) || /^[-*]\s+/.test(current) || /^\d+\.\s+/.test(current)) {
        break;
      }
      paragraphLines.push(current);
      index += 1;
    }
    blocks.push(
      <p key={`paragraph-${blocks.length}`}>
        {renderInlineMarkdown(paragraphLines.join(" "))}
      </p>,
    );
  }

  return blocks;
};

export default function FireCastBot() {
  const [config, setConfig] = useState<FireCastBotConfig | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [conversation, setConversation] = useState<ConversationEntry[]>([]);
  const [documentsCount, setDocumentsCount] = useState(0);
  const [latestTranscript, setLatestTranscript] = useState("");
  const [latestQueryClassification, setLatestQueryClassification] = useState("");
  const [queryInput, setQueryInput] = useState("");
  const [speechToTextProviderId, setSpeechToTextProviderId] = useState("");
  const [textToSpeechProviderId, setTextToSpeechProviderId] = useState("");
  const [speakResponses, setSpeakResponses] = useState(false);
  const [selectedPdfName, setSelectedPdfName] = useState("");
  const [selectedAudioName, setSelectedAudioName] = useState("");
  const [audioSrc, setAudioSrc] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSpeechPaused, setIsSpeechPaused] = useState(false);
  const [activeSpeechMessageKey, setActiveSpeechMessageKey] = useState("");
  const [error, setError] = useState("");
  const [status, setStatus] = useState("Connecting FireCastBot...");
  const [isBusy, setIsBusy] = useState(false);
  const [showSpeechSettings, setShowSpeechSettings] = useState(false);
  const pdfInputRef = useRef<HTMLInputElement | null>(null);
  const audioInputRef = useRef<HTMLInputElement | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const suppressSpeechErrorRef = useRef(false);
  const activeSpeechMessageKeyRef = useRef("");

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const [{ data: botConfig }, sessionResponse] = await Promise.all([
          axios.get<FireCastBotConfig>(`${API_BASE_URL}/api/firecastbot/config`),
          createOrResumeSession(),
        ]);
        const browserProviderId = botConfig.providers.find(provider => provider.id === "browser")?.id;
        setConfig(botConfig);
        setSpeechToTextProviderId(browserProviderId || botConfig.defaultSpeechToTextProvider);
        setTextToSpeechProviderId(browserProviderId || botConfig.defaultTextToSpeechProvider);
        applySnapshot(sessionResponse.data);
        setStatus("FireCastBot ready.");
      } catch (exc: any) {
        setError(exc?.response?.data?.error || exc?.message || "Unable to start FireCastBot.");
        setStatus("FireCastBot unavailable.");
      }
    };

    void bootstrap();

    const cancelSpeech = () => {
      if ("speechSynthesis" in window) {
        suppressSpeechErrorRef.current = true;
        window.speechSynthesis.cancel();
      }
    };

    window.addEventListener("beforeunload", cancelSpeech);
    window.addEventListener("pagehide", cancelSpeech);

    return () => {
      window.removeEventListener("beforeunload", cancelSpeech);
      window.removeEventListener("pagehide", cancelSpeech);
      cancelSpeech();
    };
  }, []);

  const createOrResumeSession = async () => {
    const existingId = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (existingId) {
      try {
        return await axios.get<SessionSnapshot>(`${API_BASE_URL}/api/firecastbot/sessions/${existingId}`);
      } catch {
        sessionStorage.removeItem(SESSION_STORAGE_KEY);
      }
    }

    const created = await axios.post<SessionSnapshot>(`${API_BASE_URL}/api/firecastbot/sessions`);
    sessionStorage.setItem(SESSION_STORAGE_KEY, created.data.sessionId);
    return created;
  };

  const createFreshSession = async () => {
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
    const created = await axios.post<SessionSnapshot>(`${API_BASE_URL}/api/firecastbot/sessions`);
    sessionStorage.setItem(SESSION_STORAGE_KEY, created.data.sessionId);
    applySnapshot(created.data);
    return created.data.sessionId;
  };

  const withSessionRetry = async <T,>(task: (activeSessionId: string) => Promise<T>) => {
    try {
      return await task(sessionId);
    } catch (exc: any) {
      if (!isUnknownSessionError(exc)) throw exc;
      const freshSessionId = await createFreshSession();
      return await task(freshSessionId);
    }
  };

  const applySnapshot = (snapshot: SessionSnapshot) => {
    setSessionId(snapshot.sessionId);
    setConversation(snapshot.conversation);
    setDocumentsCount(snapshot.documentsCount);
    setLatestTranscript(snapshot.latestTranscript);
    setLatestQueryClassification(snapshot.latestQueryClassification || "");
  };

  const getSelectedProvider = (providerId: string) =>
    config?.providers.find(provider => provider.id === providerId) || null;

  const setSpeechPlaybackState = (messageKey: string, speaking: boolean, paused: boolean) => {
    activeSpeechMessageKeyRef.current = messageKey;
    setActiveSpeechMessageKey(messageKey);
    setIsSpeaking(speaking);
    setIsSpeechPaused(paused);
  };

  const cancelBrowserSpeechInternal = () => {
    if (!("speechSynthesis" in window)) return;
    const synth = window.speechSynthesis;
    if (!utteranceRef.current && !synth.speaking && !synth.pending) {
      suppressSpeechErrorRef.current = false;
      return;
    }
    suppressSpeechErrorRef.current = true;
    synth.cancel();
    window.setTimeout(() => {
      suppressSpeechErrorRef.current = false;
    }, 0);
  };

  const startBrowserSpeech = (text: string, messageKey: string) => {
    if (!("speechSynthesis" in window) || !text.trim()) return;
    const synth = window.speechSynthesis;
    if (activeSpeechMessageKeyRef.current === messageKey) {
      if (synth.paused) {
        synth.resume();
        setSpeechPlaybackState(messageKey, true, false);
        return;
      }
      if (synth.speaking) {
        return;
      }
    }
    cancelBrowserSpeechInternal();
    const utterance = new SpeechSynthesisUtterance(text);
    utteranceRef.current = utterance;
    setSpeechPlaybackState(messageKey, true, false);
    utterance.onstart = () => {
      setSpeechPlaybackState(messageKey, true, false);
    };
    utterance.onpause = () => {
      setSpeechPlaybackState(messageKey, true, true);
    };
    utterance.onresume = () => {
      setSpeechPlaybackState(messageKey, true, false);
    };
    utterance.onend = () => {
      utteranceRef.current = null;
      setSpeechPlaybackState("", false, false);
    };
    utterance.onerror = (event: any) => {
      const errorType = String(event?.error || "").toLowerCase();
      if (
        suppressSpeechErrorRef.current ||
        errorType === "interrupted" ||
        errorType === "canceled" ||
        errorType === "cancelled"
      ) {
        suppressSpeechErrorRef.current = false;
        utteranceRef.current = null;
        setSpeechPlaybackState("", false, false);
        return;
      }
      utteranceRef.current = null;
      setSpeechPlaybackState("", false, false);
      setError("Browser speech playback failed.");
    };
    synth.speak(utterance);
  };

  const pauseBrowserSpeech = (messageKey: string) => {
    if (!("speechSynthesis" in window)) return;
    const synth = window.speechSynthesis;
    if (activeSpeechMessageKeyRef.current !== messageKey || !synth.speaking || synth.paused) return;
    synth.pause();
    setSpeechPlaybackState(messageKey, true, true);
  };

  const resumeBrowserSpeech = (messageKey: string) => {
    if (!("speechSynthesis" in window)) return;
    const synth = window.speechSynthesis;
    if (activeSpeechMessageKeyRef.current !== messageKey || !synth.paused) return;
    synth.resume();
    setSpeechPlaybackState(messageKey, true, false);
  };

  const runTask = async (task: () => Promise<void>) => {
    setIsBusy(true);
    setError("");
    try {
      await task();
    } catch (exc: any) {
      setError(exc?.response?.data?.error || exc?.message || "Request failed.");
    } finally {
      setIsBusy(false);
    }
  };

  const uploadPdf = async () => {
    const file = pdfInputRef.current?.files?.[0];
    if (!file || !sessionId) return;
    await runTask(async () => {
      setStatus("Loading PDF...");
      const { data } = await withSessionRetry(activeSessionId => {
        const formData = new FormData();
        formData.append("session_id", activeSessionId);
        formData.append("file", file);
        return axios.post(`${API_BASE_URL}/api/firecastbot/documents/pdf`, formData);
      });
      applySnapshot(data);
      setStatus(`Incident report parsed into ${data.documentsCount} runtime chunks.`);
    });
  };

  const submitQuery = async () => {
    if (!queryInput.trim() || !sessionId) return;
    await runTask(async () => {
      setStatus("Generating response...");
      const query = queryInput.trim();
      setQueryInput("");
      const { data } = await withSessionRetry(activeSessionId => axios.post(`${API_BASE_URL}/api/firecastbot/query`, {
        session_id: activeSessionId,
        query,
        speak_responses: speakResponses,
        text_to_speech_provider_id: textToSpeechProviderId,
      }));
      applySnapshot(data);
      const reply = data.reply as string;
      const latestAssistantIndex = Array.isArray(data.conversation) ? data.conversation.length - 1 : conversation.length + 1;
      if (data.audioBase64 && data.audioMimeType) {
        setAudioSrc(`data:${data.audioMimeType};base64,${data.audioBase64}`);
      } else {
        setAudioSrc("");
        const provider = getSelectedProvider(textToSpeechProviderId);
        if (provider?.outputMode === "browser" && speakResponses) {
          startBrowserSpeech(reply, `assistant-${latestAssistantIndex}`);
        }
      }
      setStatus("Response ready.");
    });
  };

  const transcribeAudio = async () => {
    const file = audioInputRef.current?.files?.[0];
    if (!file || !sessionId) return;
    await runTask(async () => {
      setStatus("Transcribing audio...");
      const { data } = await withSessionRetry(activeSessionId => {
        const formData = new FormData();
        formData.append("session_id", activeSessionId);
        formData.append("speech_to_text_provider_id", speechToTextProviderId);
        formData.append("file", file);
        return axios.post(`${API_BASE_URL}/api/firecastbot/transcribe`, formData);
      });
      applySnapshot(data);
      setQueryInput(data.transcript || "");
      setStatus("Transcript added to the question box.");
    });
  };

  const startBrowserListening = () => {
    const SpeechRecognition = getSpeechRecognition();
    if (!SpeechRecognition) {
      setError("This browser does not support speech recognition.");
      return;
    }
    setError("");
    setStatus("Listening...");
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event: any) => {
      const transcript = event.results?.[0]?.[0]?.transcript || "";
      setLatestTranscript(transcript);
      setQueryInput(transcript);
      setStatus("Transcript added to the question box.");
    };
    recognition.onerror = (event: any) => {
      setError(event?.error || "Speech recognition failed.");
      setStatus("Speech recognition failed.");
    };
    recognition.onend = () => {
      setStatus(prev => (prev === "Listening..." ? "FireCastBot ready." : prev));
    };
    recognition.start();
  };

  const selectedSttProvider = getSelectedProvider(speechToTextProviderId);
  const selectedTtsProvider = getSelectedProvider(textToSpeechProviderId);
  const showBrowserSpeechControls = selectedTtsProvider?.outputMode === "browser";
  const canSubmitQuery = !isBusy && !!queryInput.trim();
  const isBotReady = status === "FireCastBot ready.";

  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <div className={styles.heroSettings}>
          <div className={styles.settingsPanel}>
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
                  onChange={(event) => setSpeechToTextProviderId(event.target.value)}
                >
                  {config?.providers.map(provider => (
                    <option key={provider.id} value={provider.id}>{provider.label}</option>
                  ))}
                </select>

                <label className={styles.fieldLabel}>Text to speech</label>
                <select
                  value={textToSpeechProviderId}
                  onChange={(event) => setTextToSpeechProviderId(event.target.value)}
                >
                  {config?.providers.map(provider => (
                    <option key={provider.id} value={provider.id}>{provider.label}</option>
                  ))}
                </select>

                <label className={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={speakResponses}
                    onChange={(event) => setSpeakResponses(event.target.checked)}
                  />
                  Read responses aloud
                </label>

                {selectedSttProvider?.inputMode === "upload" && (
                  <>
                    <input
                      ref={audioInputRef}
                      className={styles.hiddenInput}
                      type="file"
                      accept="audio/*"
                      onChange={(event) => setSelectedAudioName(event.target.files?.[0]?.name || "")}
                    />
                    <div className={styles.filePicker}>
                      <button
                        type="button"
                        className={styles.fileTrigger}
                        onClick={() => audioInputRef.current?.click()}
                      >
                        Choose Audio
                      </button>
                      <div className={styles.fileName}>
                        {selectedAudioName || "No audio selected"}
                      </div>
                    </div>
                    <button onClick={transcribeAudio} disabled={isBusy || !sessionId}>
                      Transcribe Audio
                    </button>
                  </>
                )}

                {selectedSttProvider?.inputMode === "browser" && (
                  <button onClick={startBrowserListening} disabled={isBusy}>
                    Start Browser Listening
                  </button>
                )}

                {selectedSttProvider?.transcriptionUnavailableReason && !selectedSttProvider.transcriptionAvailable && (
                  <p className={styles.note}>{selectedSttProvider.transcriptionUnavailableReason}</p>
                )}
                {selectedTtsProvider?.synthesisUnavailableReason && !selectedTtsProvider.synthesisAvailable && (
                  <p className={styles.note}>{selectedTtsProvider.synthesisUnavailableReason}</p>
                )}
                {latestTranscript && <p className={styles.note}>Latest transcript: {latestTranscript}</p>}
              </div>
            )}
          </div>
        </div>
        <div className={styles.badge}>
          <span className={`${styles.badgeDot} ${isBotReady ? styles.badgeDotReady : ""}`} />
          <span>FIRECASTBOT</span>
        </div>
        <h1 className={styles.title}>Ask FireCastBot</h1>
        <p className={styles.subtitle}>
          Upload an incident report, extract structured facts, and query it alongside doctrine and safety references without leaving FireCastRL.
        </p>
        {error && <p className={styles.error}>{error}</p>}
      </section>

      <div className={styles.layout}>
        <aside className={styles.sidebar}>
          <div className={styles.panel}>
            <h2>Incident Report</h2>
            <label className={styles.fieldLabel}>Upload incident PDF</label>
            <input
              ref={pdfInputRef}
              className={styles.hiddenInput}
              type="file"
              accept="application/pdf"
              onChange={(event) => setSelectedPdfName(event.target.files?.[0]?.name || "")}
            />
            <div className={styles.filePicker}>
              <button
                type="button"
                className={styles.fileTrigger}
                onClick={() => pdfInputRef.current?.click()}
              >
                Upload Incident PDF
              </button>
              <div className={styles.fileName}>
                {selectedPdfName || "No file selected"}
              </div>
            </div>
            <button onClick={uploadPdf} disabled={isBusy || !sessionId}>Ingest Report</button>
          </div>

        </aside>

        <section className={styles.chatArea}>
          <div className={styles.panel}>
            <h2>Conversation</h2>
            <div className={styles.messages}>
              {conversation.length === 0 && <p className={styles.empty}>No messages yet.</p>}
              {conversation.map((entry, index) => (
                <div
                  key={`${entry.role}-${index}`}
                  className={entry.role === "assistant" ? styles.assistantMessage : styles.userMessage}
                >
                  <div className={styles.messageRole}>{entry.role}</div>
                  <div className={styles.messageBody}>
                    {entry.role === "assistant" ? renderMarkdown(entry.content) : entry.content}
                  </div>
                  {entry.role === "assistant" && showBrowserSpeechControls && (
                    <div className={styles.messageControls}>
                      <button
                        type="button"
                        className={styles.messageControlButton}
                        onClick={() => {
                          const messageKey = `${entry.role}-${index}`;
                          if (activeSpeechMessageKey === messageKey && isSpeaking && isSpeechPaused) {
                            resumeBrowserSpeech(messageKey);
                            return;
                          }
                          startBrowserSpeech(entry.content, messageKey);
                        }}
                      >
                        {activeSpeechMessageKey === `${entry.role}-${index}` && isSpeechPaused ? "Resume" : "Play"}
                      </button>
                      <button
                        type="button"
                        className={styles.messageControlButton}
                        onClick={() => pauseBrowserSpeech(`${entry.role}-${index}`)}
                        disabled={activeSpeechMessageKey !== `${entry.role}-${index}` || !isSpeaking || isSpeechPaused}
                      >
                        Pause
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className={`${styles.panel} ${styles.stickyComposer}`}>
            <h2>Chat</h2>
            <div className={styles.composerInputWrap}>
              <textarea
                value={queryInput}
                onChange={(event) => setQueryInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    if (canSubmitQuery) {
                      void submitQuery();
                    }
                  }
                }}
                placeholder="Ask about incident facts, doctrine, or strategy grounded in both..."
                rows={5}
              />
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
            {latestQueryClassification && (
              <p className={styles.note}>Latest query class: {latestQueryClassification}</p>
            )}
            {audioSrc && <audio className={styles.audio} controls src={audioSrc} />}
          </div>
        </section>
      </div>
    </div>
  );
}
