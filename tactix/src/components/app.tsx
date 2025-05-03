import { observer } from "mobx-react";
import React from "react";

import 'leaflet/dist/leaflet.css';
import { HashRouter, Link, Route, Routes } from "react-router-dom";
import Home from "../pages/Home";
import Inferno from "../pages/Inferno";
import { SimulationPage } from "../pages/SimulationPage";
import styles from "./app.module.scss";


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
