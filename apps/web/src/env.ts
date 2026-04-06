declare const __APP_CONFIG__: {
  apiBaseUrl: string;
  wsUrl: string;
  generatedAssetsBaseUrl: string;
};

export const API_BASE_URL = __APP_CONFIG__.apiBaseUrl;
export const WS_URL = __APP_CONFIG__.wsUrl;
export const GENERATED_ASSETS_BASE_URL = __APP_CONFIG__.generatedAssetsBaseUrl.replace(/\/$/, "");
