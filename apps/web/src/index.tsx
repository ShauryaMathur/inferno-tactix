import { Provider } from "mobx-react";
import { configure } from "mobx";
import React from "react";
import { createRoot } from "react-dom/client";
import { AppComponent } from "./components/app";
import { ThemeProvider } from "@mui/material/styles";
import { createStores } from "./models/stores";
import hurricanesTheme from "./material-ui-theme";
import L from 'leaflet';
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl:        markerIcon,
  shadowUrl:      markerShadow,
});


// Disable mobx strict mode. Make v6 compatible with v4/v5 that was not enforcing strict mode by default.
configure({ enforceActions: "never", safeDescriptors: false });

const stores = createStores();

const container = document.getElementById("app");

if (container) {
  const root = createRoot(container);
  root.render(
    <Provider stores={stores}>
      <ThemeProvider theme={hurricanesTheme}>
        <AppComponent />
      </ThemeProvider>
    </Provider>
  );
}
