import { observer } from "mobx-react";
import React from "react";

import 'leaflet/dist/leaflet.css';
import { HashRouter, Link, Route, Routes, Navigate  } from "react-router-dom";
// import Home from "../pages/Home";
import Inferno from "../pages/inferno/Inferno";
import { SimulationPage } from "../pages/SimulationPage";
import styles from "./app.module.scss";
import About from "../pages/About";

export const AppComponent = observer(function WrappedComponent() {

  return (
    <HashRouter>
      <div id={styles.app}>

        <div className={styles.links}>
          {/* <Link to="/">Home</Link> */}
          <Link to="/about">Home</Link>
          <Link to="/inferno">Inferno</Link>
          <Link to="/tactics">Tactics</Link>
        </div>


        <Routes>
          <Route path="/" element={<Navigate replace to="/about" />} />
          {/* <Route path="/" element={<Home />} /> */}
          <Route path="/inferno" element={<Inferno />} />
          <Route path="/tactics" element={<SimulationPage />} />
          <Route path="/about" element={<About />} />
          {/* add more routes as you like */}
        </Routes>
      </div></HashRouter>
  );
});
