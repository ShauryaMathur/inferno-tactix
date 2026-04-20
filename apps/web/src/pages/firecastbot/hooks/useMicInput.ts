import { useRef, useState } from "react";
import axios from "axios";
import { API_BASE_URL } from "../../../env";
import type { Provider, SessionSnapshot } from "../types";

const getSpeechRecognition = () =>
  (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

export function useMicInput({
  speechToTextProviderId,
  isBotReady,
  isBusy,
  getSelectedProvider,
  onError,
  setQueryInput,
  runTask,
  withSessionRetry,
  applySnapshot,
}: {
  speechToTextProviderId: string;
  isBotReady: boolean;
  isBusy: boolean;
  getSelectedProvider: (id: string) => Provider | null;
  onError: (msg: string) => void;
  setQueryInput: (v: string) => void;
  runTask: (task: () => Promise<void>) => Promise<void>;
  withSessionRetry: <T>(task: (id: string) => Promise<T>) => Promise<T>;
  applySnapshot: (snapshot: SessionSnapshot) => void;
}) {
  const [isListening, setIsListening] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recognitionRef = useRef<any>(null);

  const startBrowserListening = () => {
    const SpeechRecognition = getSpeechRecognition();
    if (!SpeechRecognition) {
      onError("This browser does not support speech recognition.");
      return;
    }
    // Stop any in-flight recognition before starting a new one
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    setIsListening(true);
    const recognition = new SpeechRecognition();
    recognitionRef.current = recognition;
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event: any) => {
      const transcript = event.results?.[0]?.[0]?.transcript || "";
      setQueryInput(transcript);
    };
    recognition.onerror = (event: any) => onError(event?.error || "Speech recognition failed.");
    recognition.onend = () => {
      recognitionRef.current = null;
      setIsListening(false);
    };
    recognition.start();
  };

  const startServerListening = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      onError("Microphone access is not supported in this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        setIsRecording(false);
        await runTask(async () => {
          const { data } = await withSessionRetry((id) => {
            const formData = new FormData();
            formData.append("session_id", id);
            formData.append("speech_to_text_provider_id", speechToTextProviderId);
            formData.append("file", blob, "recording.webm");
            return axios.post(`${API_BASE_URL}/api/firecastbot/transcribe`, formData);
          });
          applySnapshot(data);
          setQueryInput(data.transcript || "");
        });
      };

      recorder.start();
      setIsRecording(true);
    } catch (err: any) {
      onError(err?.message || "Could not access microphone.");
    }
  };

  const stopServerListening = () => mediaRecorderRef.current?.stop();

  const handleMicClick = () => {
    const provider = getSelectedProvider(speechToTextProviderId);
    if (provider?.inputMode === "browser") {
      startBrowserListening();
    } else if (isRecording) {
      stopServerListening();
    } else {
      void startServerListening();
    }
  };

  return { isListening, isRecording, handleMicClick };
}
