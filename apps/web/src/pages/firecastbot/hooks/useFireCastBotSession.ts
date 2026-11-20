import axios from 'axios';
import { useEffect, useState } from 'react';
import { API_BASE_URL } from '../../../env';

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const isRetryableError = (exc: any): boolean => {
  const status = exc?.response?.status;
  if (!status) return true; // network error / timeout
  return status >= 500 && status < 600;
};
import type { ConversationEntry, FireCastBotConfig, Provider, SessionSnapshot } from '../types';

const SESSION_STORAGE_KEY = 'firecastbot-session-id';

const isUnknownSessionError = (exc: any) =>
  exc?.response?.status === 404 &&
  typeof exc?.response?.data?.error === 'string' &&
  exc.response.data.error.includes('Unknown FireCastBot session');

export function useFireCastBotSession({
  queryInput,
  setQueryInput,
  pdfInputRef,
  onNewReply,
  cancelBrowserSpeech,
  closeSettings,
}: {
  queryInput: string;
  setQueryInput: (v: string) => void;
  pdfInputRef: React.RefObject<HTMLInputElement | null>;
  onNewReply: (reply: string, assistantIndex: number) => void;
  cancelBrowserSpeech: () => void;
  closeSettings: () => void;
}) {
  const [config, setConfig] = useState<FireCastBotConfig | null>(null);
  const [sessionId, setSessionId] = useState('');
  const [conversation, setConversation] = useState<ConversationEntry[]>([]);
  const [documentsCount, setDocumentsCount] = useState(0);
  const [selectedPdfName, setSelectedPdfName] = useState('');
  const [isBusy, setIsBusy] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const [sessionReadyAt, setSessionReadyAt] = useState<Date | null>(null);

  const isBotReady = !error && documentsCount > 0;
  const isIncidentSourceLocked = documentsCount > 0;

  const applySnapshot = (snapshot: SessionSnapshot) => {
    setSessionId(snapshot.sessionId);
    setConversation(snapshot.conversation);
    setDocumentsCount((prev) => {
      if (prev === 0 && snapshot.documentsCount > 0) {
        setSessionReadyAt(new Date());
      }
      return snapshot.documentsCount;
    });
  };

  const getSelectedProvider = (providerId: string): Provider | null =>
    config?.providers.find((p) => p.id === providerId) ?? null;

  const runTask = async (task: () => Promise<void>) => {
    setIsBusy(true);
    setError('');
    try {
      await task();
    } catch (exc: any) {
      setError(exc?.response?.data?.error || exc?.message || 'Request failed.');
    } finally {
      setIsBusy(false);
    }
  };

  const createOrResumeSession = async () => {
    const existingId = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (existingId) {
      try {
        return await axios.get<SessionSnapshot>(
          `${API_BASE_URL}/api/firecastbot/sessions/${existingId}`
        );
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

  const withSessionRetry = async <T>(task: (id: string) => Promise<T>): Promise<T> => {
    try {
      return await task(sessionId);
    } catch (exc: any) {
      if (!isUnknownSessionError(exc)) throw exc;
      const freshSessionId = await createFreshSession();
      return await task(freshSessionId);
    }
  };

  // Bootstrap runs on mount and on explicit retry
  useEffect(() => {
    const bootstrap = async () => {
      setIsLoading(true);
      setError('');
      try {
        const [{ data: botConfig }, sessionResponse] = await Promise.all([
          axios.get<FireCastBotConfig>(`${API_BASE_URL}/api/firecastbot/config`),
          createOrResumeSession(),
        ]);
        setConfig(botConfig);
        applySnapshot(sessionResponse.data);
      } catch (exc: any) {
        setError(exc?.response?.data?.error || exc?.message || 'Unable to start FireCastBot.');
      } finally {
        setIsLoading(false);
      }
    };
    void bootstrap();
  }, [retryCount]); // eslint-disable-line react-hooks/exhaustive-deps

  const retryBootstrap = () => setRetryCount((c) => c + 1);

  const uploadPdf = async () => {
    const file = pdfInputRef.current?.files?.[0];
    if (!file || !sessionId) return;
    await runTask(async () => {
      const { data } = await withSessionRetry((id) => {
        const formData = new FormData();
        formData.append('session_id', id);
        formData.append('file', file);
        return axios.post(`${API_BASE_URL}/api/firecastbot/documents/pdf`, formData);
      });
      applySnapshot(data);
    });
  };

  const ingestPreset = async (presetId: string, presetLabel: string) => {
    if (!sessionId) return;
    await runTask(async () => {
      const { data } = await withSessionRetry((id) =>
        axios.post(`${API_BASE_URL}/api/firecastbot/documents/preset`, {
          session_id: id,
          preset_id: presetId,
        })
      );
      applySnapshot(data);
      setSelectedPdfName(presetLabel);
    });
  };

  const submitQueryText = async (query: string) => {
    if (!query.trim() || !sessionId) return;
    setIsQuerying(true);
    try {
      await runTask(async () => {
        const MAX_ATTEMPTS = 3;
        let lastError: any;
        for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
          try {
            const { data } = await axios.post(`${API_BASE_URL}/api/firecastbot/query`, {
              session_id: sessionId,
              query: query.trim(),
            });
            applySnapshot(data);
            const reply = data.reply as string;
            const latestAssistantIndex = data.conversation.length - 1;
            onNewReply(reply, latestAssistantIndex);
            return;
          } catch (exc: any) {
            if (isUnknownSessionError(exc)) {
              throw new Error('Your session has expired. Please reload an incident report to continue.');
            }
            if (exc?.response?.status === 429) {
              const retryAfter = exc.response.data?.retryAfter ?? 1;
              throw new Error(`Too many requests — please wait ${Math.ceil(retryAfter)} second(s) and try again.`);
            }
            if (!isRetryableError(exc) || attempt === MAX_ATTEMPTS) throw exc;
            lastError = exc;
            await sleep(attempt * 1000); // 1s then 2s back-off
          }
        }
        throw lastError;
      });
    } finally {
      setIsQuerying(false);
    }
  };

  const submitQuery = async () => {
    if (!queryInput.trim()) return;
    const query = queryInput.trim();
    setQueryInput('');
    await submitQueryText(query);
  };

  const startNewSession = async () => {
    await runTask(async () => {
      cancelBrowserSpeech();
      closeSettings();
      await createFreshSession();
      // Reset UI state only after the new session is confirmed
      setSelectedPdfName('');
      setQueryInput('');
      setSessionReadyAt(null);
      if (pdfInputRef.current) pdfInputRef.current.value = '';
    });
  };

  return {
    config,
    sessionId,
    conversation,
    documentsCount,
    selectedPdfName,
    setSelectedPdfName,
    isBusy,
    isQuerying,
    isLoading,
    error,
    setError,
    retryBootstrap,
    isBotReady,
    isIncidentSourceLocked,
    sessionReadyAt,
    getSelectedProvider,
    runTask,
    withSessionRetry,
    applySnapshot,
    uploadPdf,
    ingestPreset,
    submitQuery,
    submitQueryText,
    startNewSession,
  };
}
