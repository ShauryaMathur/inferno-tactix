import React, { useEffect } from "react";
import { observer } from "mobx-react";
import { View3d } from "./view-3d/view-3d";
import { SimulationInfo } from "./simulation-info";
import { TerrainPanel } from "./terrain-panel";
import { RightPanel } from "./right-panel";
import { BottomBar } from "./bottom-bar";
import { useStores } from "../use-stores";
import { TopBar } from "./top-bar/top-bar";
import { AboutDialogContent } from "./top-bar/about-dialog-content";
import { ShareDialogContent } from "./top-bar/share-dialog-content";
import Shutterbug from "shutterbug";

import styles from "./app.module.scss";
import { useCustomCursor } from "./use-custom-cursors";
import { BrowserRouter, HashRouter, Link, Route, Routes } from "react-router-dom";
import { SimulationPage } from "../pages/SimulationPage";
import Home from "../pages/Home";
import 'leaflet/dist/leaflet.css';
import Inferno from "../pages/Inferno";


export const AppComponent = observer(function WrappedComponent() {

  return (
    <HashRouter>
      <div className={styles.homePage}>

        <div className={styles.links}>
        <Link to="/">Home</Link>
          <Link to="/inferno">Inferno</Link>
          <Link to="/tactics">Tactics</Link>
        </div>


        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/inferno" element={<Inferno />} />
          <Route path="/tactics" element={<SimulationPage />} />
          {/* add more routes as you like */}
        </Routes>
      </div></HashRouter>
  );
});
