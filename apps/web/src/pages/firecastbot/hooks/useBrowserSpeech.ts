import { useRef, useState } from "react";

export function useBrowserSpeech(onError: (msg: string) => void) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSpeechPaused, setIsSpeechPaused] = useState(false);
  const [activeSpeechMessageKey, setActiveSpeechMessageKey] = useState("");
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const suppressSpeechErrorRef = useRef(false);
  const activeSpeechMessageKeyRef = useRef("");

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
    window.setTimeout(() => suppressSpeechErrorRef.current = false, 0);
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
      if (synth.speaking) return;
    }
    cancelBrowserSpeechInternal();
    const utterance = new SpeechSynthesisUtterance(text);
    utteranceRef.current = utterance;
    setSpeechPlaybackState(messageKey, true, false);
    utterance.onstart = () => setSpeechPlaybackState(messageKey, true, false);
    utterance.onpause = () => setSpeechPlaybackState(messageKey, true, true);
    utterance.onresume = () => setSpeechPlaybackState(messageKey, true, false);
    utterance.onend = () => {
      utteranceRef.current = null;
      setSpeechPlaybackState("", false, false);
    };
    utterance.onerror = (event: any) => {
      const errorType = String(event?.error || "").toLowerCase();
      if (suppressSpeechErrorRef.current || errorType === "interrupted" || errorType === "canceled" || errorType === "cancelled") {
        suppressSpeechErrorRef.current = false;
        utteranceRef.current = null;
        setSpeechPlaybackState("", false, false);
        return;
      }
      utteranceRef.current = null;
      setSpeechPlaybackState("", false, false);
      onError("Browser speech playback failed.");
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

  return {
    isSpeaking,
    isSpeechPaused,
    activeSpeechMessageKey,
    startBrowserSpeech,
    pauseBrowserSpeech,
    resumeBrowserSpeech,
    cancelBrowserSpeechInternal,
  };
}
