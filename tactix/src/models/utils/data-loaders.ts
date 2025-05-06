import { TerrainType } from "../../types";
import { ISimulationConfig } from "../../config";
import { getInputData } from "./image-utils";
import { Zone } from "../zone";
import { log } from "console";

// Maps zones config to image data files (see data dir)
const zonesToImageDataFile = (zones: Zone[]): string => {
  const zoneTypes = zones.map(z => TerrainType[z.terrainType].toLowerCase());
  return `data/${zoneTypes.join("-")}`;
};

export const getZoneIndex = (
  config: ISimulationConfig,
  zoneIndex: number[][] | string
): Promise<number[] | undefined> => {
  return getInputData(
    zoneIndex,
    config.gridWidth,
    config.gridHeight,
    false,
    (rgba: [number, number, number, number]) => {
      // Red is zone 1, green is zone 2, blue is zone 3
      if (rgba[0] >= rgba[1] && rgba[0] >= rgba[2]) return 0;
      if (rgba[1] >= rgba[0] && rgba[1] >= rgba[2]) return 1;
      return 2;
    }
  );
};

export const getElevationData = (
  config: ISimulationConfig,
  zones: Zone[]
): Promise<number[] | undefined> => {
  // Determine elevation PNG path
  let elevation = config.elevation;
  console.log(elevation);
  
  if (!elevation) {
    // elevation = `${zonesToImageDataFile(zones)}-heightmap.png`;
    // elevation = "data/heightmap_1200x800.png";
    // elevation = "data/heightmap_1200x800_2.png";
    elevation = "data/heightmap_1200x813_2.png";
  }

  // Decode a 16-bit RGBA heightmap where R=high byte, G=low byte
  const heightFn = (rgba: [number, number, number, number]) => {
    const highByte = rgba[0];
    const lowByte = rgba[1];
    const value16 = (highByte << 8) | lowByte;      // 0–65535
    const hNorm = value16 / 65535;                  // normalized
    return hNorm * config.heightmapMaxElevation;    // scale to meters
  };

  return getInputData(
    elevation,
    config.gridWidth,
    config.gridHeight,
    true,
    heightFn
  );
};

export const getUnburntIslandsData = (
  config: ISimulationConfig,
  zones: Zone[]
): Promise<number[] | undefined> => {
  // Determine islands image path
  let islandsFile = config.unburntIslands;
  if (!islandsFile) {
    islandsFile = `${zonesToImageDataFile(zones)}-islands.png`;
  }

  const islandActive: Record<number, number> = {};
  return getInputData(
    islandsFile,
    config.gridWidth,
    config.gridHeight,
    true,
    (rgba: [number, number, number, number]) => {
      const shade = rgba[0];
      if (shade < 255) {
        if (islandActive[shade] === undefined) {
          islandActive[shade] = Math.random() < config.unburntIslandProbability ? 1 : 0;
        }
        return islandActive[shade];
      }
      return 0;
    }
  );
};

export const getRiverData = (
  config: ISimulationConfig
): Promise<number[] | undefined> => {
  if (!config.riverData) return Promise.resolve(undefined);

  return getInputData(
    config.riverData,
    config.gridWidth,
    config.gridHeight,
    true,
    (rgba: [number, number, number, number]) => (rgba[3] > 0 ? 1 : 0)
  );
};
