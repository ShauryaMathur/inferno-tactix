export type Provider = {
  id: string;
  label: string;
  transcriptionAvailable: boolean;
  transcriptionUnavailableReason: string | null;
  inputMode: 'upload' | 'browser' | 'none';
  synthesisAvailable: boolean;
  synthesisUnavailableReason: string | null;
  outputMode: 'audio_bytes' | 'browser' | 'none';
};

export type FireCastBotConfig = {
  defaultSpeechToTextProvider: string;
  defaultTextToSpeechProvider: string;
  providers: Provider[];
  presets: Preset[];
};

export type Preset = {
  id: string;
  label: string;
  available: boolean;
  previewUrl?: string;
};

export type ConversationEntry = {
  role: string;
  content: string;
};

export type SessionSnapshot = {
  sessionId: string;
  documentsCount: number;
  conversation: ConversationEntry[];
  latestTranscript: string;
};
