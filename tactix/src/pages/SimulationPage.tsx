import { observer } from "mobx-react";
import React, { useEffect } from "react";
import Shutterbug from "shutterbug";
import { TerrainPanel } from "../components/terrain-panel";
import { View3d } from "../components/view-3d/view-3d";
import { useStores } from "../use-stores";

import { useSearchParams } from "react-router-dom";
import css from "../components/app.module.scss";
import { useCustomCursor } from "../components/use-custom-cursors";

export const SimulationPage = observer(function WrappedComponent() {
  const { simulation, ui } = useStores();
  const [searchParams] = useSearchParams();

  const lat = parseFloat(searchParams.get('lat') ?? '37.8');
  const lon = parseFloat(searchParams.get('lon') ?? '-96.0');
  const date = searchParams.get('date') ?? new Date().toISOString().slice(0,10);


  useEffect(() => {
    Shutterbug.enable("." + css.app);
    return () => {
      Shutterbug.disable();
    };
  }, []);

  // This will setup document cursor based on various states of UI store (interactions).
  useCustomCursor();

  const config = simulation.config;
  // Convert time from minutes to days.
  const timeInDays = Math.floor(simulation.time / 1440);
  const timeHours = Math.floor((simulation.time % 1440) / 60);
  const showModelScale = config.showModelDimensions;
  const episodeCount = simulation.episodeCount;
  return (
    <div className={css.app}>
      {/* <TopBar projectName="Wildfire Explorer" aboutContent={<AboutDialogContent />} shareContent={<ShareDialogContent />} /> */}
      { showModelScale &&
        <div className={css.modelInfo}>
          <div>Model Dimensions: { config.modelWidth } ft x { config.modelHeight } ft</div>
          <div>Highest Point Possible: {config.heightmapMaxElevation} ft</div>
        </div>
      }
      <div className={css.timeDisplay} style={{height:'auto'}}>
       Episode : {episodeCount} <br /> {timeInDays} {timeInDays === 1 ? "day" : "days"} and <br /> {timeHours} {timeHours === 1 ? "hour" : "hours"}
      </div>
      <div className={`${css.mainContent} ${ui.showChart && css.shrink}`}>
        {/* <SimulationInfo /> */}
        <View3d />
        <TerrainPanel />
      </div>
      {/* <div className={`${css.rightContent} ${ui.showChart && css.grow}`}>
        <RightPanel />
      </div> */}
      {/* <BottomBar /> */}
    </div>
  );
});
